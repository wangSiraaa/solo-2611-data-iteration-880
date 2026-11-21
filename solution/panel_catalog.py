"""Guide panels with their own revisions, projecting positions onto paper.

This module depends on :mod:`chainage_catalog` only -- never on reports or
the typesetting bench.  It is the *catalog side* of the print room: it
delivers annotation anchors and the projection basis independently, so an
old panel's proof can be re-checked offline against the retained revisions.

Model
-----
A **panel** (``panel_id``) has its own revision sequence (``panel_revision``)
that is independent of the route catalog revisions.  A panel revision is an
ordered set of named **strips**.  Each strip binds:

* a route catalog revision (must already exist),
* a section within it,
* a legal half-open label window inside that section (validated with the
  catalog's own exact span conversion -- upper bound may equal the
  section's exclusive end label, as elsewhere),
* two *distinct* paper x-coordinates for the two window ends.  Window
  start maps to ``paper_start`` and window end to ``paper_end``;
  ``paper_start < paper_end`` is forward (顺排), ``paper_start >
  paper_end`` is reverse (逆排).

Projection is affine and exact (Fraction arithmetic): a physical position
``p`` inside the window maps to

    paper(p) = paper_start + (p - phys_lo) * (paper_end - paper_start)
               / (phys_hi - phys_lo)

where ``[phys_lo, phys_hi)`` is the window's physical interval from the
bound catalog revision.  Because the catalog's label slope is 1 this is
equally the proportional map of the *label* window; physical positions are
used so a label-base revision can never silently move an anchor.

Immutability: a ``(panel_id, panel_revision)`` pair is content-addressed.
Re-defining identical content is idempotent; different content under the
same key is rejected.  Old revisions are retained forever, which is what
makes offline re-checks of old proofs possible.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

from chainage_catalog import (
    CatalogLookupError,
    CatalogRegistry,
    RouteCatalog,
    as_fraction,
    fraction_to_db,
)


class PanelError(ValueError):
    """A panel/strip definition or anchor request is invalid."""


class PanelConflictError(PanelError):
    """A (panel_id, panel_revision) key was reused with different content."""


class OutsideWindowError(PanelError):
    """A physical position does not lie in the strip's window."""


# ---------------------------------------------------------------------------
# Canonical JSON helpers (exact rationals as 'n/d' text)
# ---------------------------------------------------------------------------


def canonical(obj):
    """Recursively convert Fractions to 'n/d' text for canonical JSON."""
    if isinstance(obj, Fraction):
        return fraction_to_db(obj)
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    return obj


def fingerprint_of(obj) -> str:
    payload = json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Strips and panel revisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Strip:
    strip_id: str
    catalog_revision: str
    section_id: str
    label_start: Fraction
    label_end: Fraction
    paper_start: Fraction  # paper x of the window *start*
    paper_end: Fraction  # paper x of the window *end* (must differ)

    @property
    def reversed(self) -> bool:
        """逆排: window start sits to the *right* of window end on paper."""
        return self.paper_end < self.paper_start

    def to_canonical(self) -> dict:
        return {
            "strip_id": self.strip_id,
            "catalog_revision": self.catalog_revision,
            "section_id": self.section_id,
            "label_start": self.label_start,
            "label_end": self.label_end,
            "paper_start": self.paper_start,
            "paper_end": self.paper_end,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Strip":
        return cls(
            strip_id=str(data["strip_id"]),
            catalog_revision=str(data["catalog_revision"]),
            section_id=str(data["section_id"]),
            label_start=as_fraction(data["label_start"]),
            label_end=as_fraction(data["label_end"]),
            paper_start=as_fraction(data["paper_start"]),
            paper_end=as_fraction(data["paper_end"]),
        )


@dataclass(frozen=True)
class PanelRevision:
    panel_id: str
    panel_revision: str
    strips: tuple[Strip, ...]

    def strip_or_none(self, strip_id: str) -> Optional[Strip]:
        for s in self.strips:
            if s.strip_id == strip_id:
                return s
        return None

    def strip(self, strip_id: str) -> Strip:
        s = self.strip_or_none(strip_id)
        if s is None:
            raise PanelError(
                f"panel {self.panel_id}/{self.panel_revision} has no strip "
                f"{strip_id!r}"
            )
        return s

    def to_canonical(self) -> dict:
        return {
            "panel_id": self.panel_id,
            "panel_revision": self.panel_revision,
            "strips": [s.to_canonical() for s in self.strips],
        }

    def fingerprint(self) -> str:
        return fingerprint_of(self.to_canonical())


class PanelCatalog:
    """Versioned panels on top of a route :class:`CatalogRegistry`."""

    def __init__(self, registry: CatalogRegistry):
        self.registry = registry
        self._revisions: dict[tuple[str, str], PanelRevision] = {}
        self._order: list[tuple[str, str]] = []  # creation order

    # -- definition ----------------------------------------------------
    def define_panel(
        self, panel_id: str, panel_revision: str, strips: Sequence[Strip]
    ) -> PanelRevision:
        """Define (or idempotently re-define) one panel revision.

        Every strip is validated against its bound catalog revision *now*:
        the revision must exist, the section must exist, the label window
        must be legal for that section, and the two paper points must
        differ.  Nothing is stored until all strips validate.
        """
        if not panel_id or not panel_revision:
            raise PanelError("panel_id and panel_revision must be non-empty")
        if not strips:
            raise PanelError(f"panel {panel_id}/{panel_revision} has no strips")
        seen = set()
        validated = []
        for strip in strips:
            if not isinstance(strip, Strip):
                strip = Strip.from_dict(strip)
            if strip.strip_id in seen:
                raise PanelError(
                    f"duplicate strip_id {strip.strip_id!r} in panel "
                    f"{panel_id}/{panel_revision}"
                )
            seen.add(strip.strip_id)
            self._validate_strip(strip)
            validated.append(strip)
        prev = PanelRevision(panel_id, panel_revision, tuple(validated))
        key = (panel_id, panel_revision)
        existing = self._revisions.get(key)
        if existing is not None:
            if existing.fingerprint() != prev.fingerprint():
                raise PanelConflictError(
                    f"panel {panel_id}/{panel_revision} already exists with "
                    f"different content"
                )
            return existing
        self._revisions[key] = prev
        self._order.append(key)
        return prev

    def _validate_strip(self, strip: Strip) -> None:
        try:
            catalog = self.registry.get(strip.catalog_revision)
        except CatalogLookupError as exc:
            raise PanelError(
                f"strip {strip.strip_id!r} binds unknown catalog revision "
                f"{strip.catalog_revision!r}"
            ) from exc
        if strip.paper_start == strip.paper_end:
            raise PanelError(
                f"strip {strip.strip_id!r} maps both window ends to paper "
                f"{fraction_to_db(strip.paper_start)}; the two paper points "
                f"must differ"
            )
        try:
            # The catalog's own exact conversion validates the window.
            catalog.convert_span(
                strip.section_id, strip.label_start, strip.label_end
            )
        except (CatalogLookupError, ValueError) as exc:
            raise PanelError(
                f"strip {strip.strip_id!r} has no legal label window: {exc}"
            ) from exc

    # -- reads -----------------------------------------------------------
    def get(self, panel_id: str, panel_revision: Optional[str] = None) -> PanelRevision:
        if panel_revision is None:
            keys = [k for k in self._order if k[0] == panel_id]
            if not keys:
                raise PanelError(f"unknown panel {panel_id!r}")
            return self._revisions[keys[-1]]
        try:
            return self._revisions[(panel_id, panel_revision)]
        except KeyError as exc:
            raise PanelError(
                f"unknown panel revision {panel_id}/{panel_revision}"
            ) from exc

    def revisions(self, panel_id: str) -> list[PanelRevision]:
        return [self._revisions[k] for k in self._order if k[0] == panel_id]

    # -- anchors (the catalog-side delivery) -----------------------------
    def anchor(
        self,
        panel_id: str,
        strip_id: str,
        physical,
        panel_revision: Optional[str] = None,
    ) -> dict:
        """Deliver the annotation anchor and its projection basis.

        ``physical`` is the facility's fixed physical position.  The result
        carries everything a proof needs for an offline re-check: the bound
        catalog revision, section, label and physical windows, paper
        endpoints, direction and the exact paper-per-physical scale.
        """
        prev = self.get(panel_id, panel_revision)
        strip = prev.strip(strip_id)
        catalog = self.registry.get(strip.catalog_revision)
        conv = catalog.convert_span(
            strip.section_id, strip.label_start, strip.label_end
        )
        iv = conv["physical"]
        p = as_fraction(physical)
        if not (iv.start <= p < iv.end):  # half-open, as everywhere
            raise OutsideWindowError(
                f"physical {fraction_to_db(p)} is outside strip "
                f"{strip.strip_id!r} window "
                f"[{fraction_to_db(iv.start)}, {fraction_to_db(iv.end)})"
            )
        scale = (strip.paper_end - strip.paper_start) / (iv.end - iv.start)
        x = strip.paper_start + (p - iv.start) * scale
        section = catalog.section(strip.section_id)
        return {
            "panel_id": prev.panel_id,
            "panel_revision": prev.panel_revision,
            "strip_id": strip.strip_id,
            "physical": p,
            "anchor": x,
            "basis": {
                "catalog_revision": strip.catalog_revision,
                "section_id": strip.section_id,
                "label_window": (strip.label_start, strip.label_end),
                "physical_window": (iv.start, iv.end),
                "label_at_physical": section.label_at(p),
                "paper_start": strip.paper_start,
                "paper_end": strip.paper_end,
                "reversed": strip.reversed,
                "scale": scale,  # paper units per physical unit (signed)
            },
        }


# ---------------------------------------------------------------------------
# SQLite persistence (separate schema from catalog and inspection facts)
# ---------------------------------------------------------------------------


PANEL_SCHEMA = """
CREATE TABLE IF NOT EXISTS panel_revision (
    panel_id       TEXT NOT NULL,
    panel_revision TEXT NOT NULL,
    fingerprint    TEXT NOT NULL,
    created_seq    INTEGER NOT NULL UNIQUE,
    PRIMARY KEY (panel_id, panel_revision)
);
CREATE TABLE IF NOT EXISTS panel_strip (
    panel_id        TEXT NOT NULL,
    panel_revision  TEXT NOT NULL,
    strip_id        TEXT NOT NULL,
    strip_order     INTEGER NOT NULL,
    catalog_revision TEXT NOT NULL,
    section_id      TEXT NOT NULL,
    label_start_num TEXT NOT NULL, label_start_den TEXT NOT NULL,
    label_end_num   TEXT NOT NULL, label_end_den   TEXT NOT NULL,
    paper_start_num TEXT NOT NULL, paper_start_den TEXT NOT NULL,
    paper_end_num   TEXT NOT NULL, paper_end_den   TEXT NOT NULL,
    PRIMARY KEY (panel_id, panel_revision, strip_id)
);
"""


def ensure_panel_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(PANEL_SCHEMA)


def persist_panel_revision(conn: sqlite3.Connection, prev: PanelRevision) -> None:
    """Store a panel revision; identical content is idempotent."""
    with conn:
        row = conn.execute(
            "SELECT fingerprint FROM panel_revision "
            "WHERE panel_id = ? AND panel_revision = ?",
            (prev.panel_id, prev.panel_revision),
        ).fetchone()
        if row is not None:
            if row["fingerprint"] != prev.fingerprint():
                raise PanelConflictError(
                    f"panel {prev.panel_id}/{prev.panel_revision} stored "
                    f"with different content"
                )
            return
        seq = conn.execute(
            "SELECT COALESCE(MAX(created_seq), 0) + 1 FROM panel_revision"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO panel_revision (panel_id, panel_revision, fingerprint, "
            "created_seq) VALUES (?, ?, ?, ?)",
            (prev.panel_id, prev.panel_revision, prev.fingerprint(), seq),
        )
        for order, s in enumerate(prev.strips):
            conn.execute(
                "INSERT INTO panel_strip (panel_id, panel_revision, strip_id, "
                "strip_order, catalog_revision, section_id, "
                "label_start_num, label_start_den, label_end_num, label_end_den, "
                "paper_start_num, paper_start_den, paper_end_num, paper_end_den) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    prev.panel_id, prev.panel_revision, s.strip_id, order,
                    s.catalog_revision, s.section_id,
                    str(s.label_start.numerator), str(s.label_start.denominator),
                    str(s.label_end.numerator), str(s.label_end.denominator),
                    str(s.paper_start.numerator), str(s.paper_start.denominator),
                    str(s.paper_end.numerator), str(s.paper_end.denominator),
                ),
            )


def load_panel_revisions(conn: sqlite3.Connection) -> list[PanelRevision]:
    """Load all panel revisions in creation order."""
    heads = conn.execute(
        "SELECT * FROM panel_revision ORDER BY created_seq"
    ).fetchall()
    out = []
    for h in heads:
        strips = [
            Strip(
                strip_id=r["strip_id"],
                catalog_revision=r["catalog_revision"],
                section_id=r["section_id"],
                label_start=Fraction(
                    int(r["label_start_num"]), int(r["label_start_den"])
                ),
                label_end=Fraction(
                    int(r["label_end_num"]), int(r["label_end_den"])
                ),
                paper_start=Fraction(
                    int(r["paper_start_num"]), int(r["paper_start_den"])
                ),
                paper_end=Fraction(
                    int(r["paper_end_num"]), int(r["paper_end_den"])
                ),
            )
            for r in conn.execute(
                "SELECT * FROM panel_strip WHERE panel_id = ? AND "
                "panel_revision = ? ORDER BY strip_order",
                (h["panel_id"], h["panel_revision"]),
            ).fetchall()
        ]
        out.append(
            PanelRevision(h["panel_id"], h["panel_revision"], tuple(strips))
        )
    return out


def load_panel_catalog(
    conn: sqlite3.Connection, registry: CatalogRegistry
) -> PanelCatalog:
    """Rebuild a PanelCatalog from storage, re-validating every strip."""
    panels = PanelCatalog(registry)
    for prev in load_panel_revisions(conn):
        panels.define_panel(prev.panel_id, prev.panel_revision, prev.strips)
    return panels
