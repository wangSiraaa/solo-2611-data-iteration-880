"""Press room: annotation sheets, proof sheets, line issues, plate freezes.

This module combines the two independent sides:

* :mod:`panel_catalog` -- the catalog side, which delivers annotation
  anchors and the projection basis on its own;
* :mod:`inspection_library` -- the report side, which records which
  facilities a historical report actually covered;
* :mod:`typeset_bench` -- the pure line-layout optimiser.

An **annotation sheet** (标注单) names a report, a panel revision, and
entries ``(note_id, strip_id, asset_id)``.  Every entry is checked against
the *facts*: the strip must exist in that panel revision, the strip's
bound catalog revision must equal the report's revision, the facility must
be covered by the report, and its physical position must lie in the
strip's window.  Each failure is recorded with a concrete reason; accepted
entries get their anchor from the catalog side.  Sheets are
content-addressed like reports, and a stored sheet can be **re-checked
offline** (``verify_sheet``): the proof is rebuilt from the retained
report, catalog and panel revisions and the fingerprints are compared.

A **line issue** is one typesetting of a line spec against a sheet's
anchors under a specific panel revision.  A **plate** freezes a batch of
line issues (印版每天批量冻结).  When the paper changes slightly (a new
panel revision), ``reissue_plate`` re-typesets only the lines whose
anchors actually moved; unaffected lines carry their frozen issue into the
new plate.  ``reprint_sheet``/``verify_sheet`` (复印旧单) and
``typeset_line`` (新排版) are the public operations.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional, Sequence

from chainage_catalog import as_fraction, fraction_to_db
from inspection_library import InspectionLibrary
from panel_catalog import (
    OutsideWindowError,
    PanelCatalog,
    PanelError,
    canonical,
    fingerprint_of,
)
from typeset_bench import (
    LineItem,
    LineSpec,
    LineSpecError,
    Placement,
    typeset as bench_typeset,
)


class SheetError(ValueError):
    """An annotation sheet cannot be issued at all."""


class SheetConflictError(SheetError):
    """A sheet_id was reused with different content."""


class LineConflictError(ValueError):
    """A line_id was reused with a different spec."""


class PlateError(ValueError):
    """A plate operation is invalid."""


class PlateConflictError(PlateError):
    """A plate_id was reused with different content."""


# Concrete per-entry rejection reasons.
REASON_UNKNOWN_STRIP = "unknown_strip"
REASON_REVISION_MISMATCH = "revision_mismatch"
REASON_FACILITY_NOT_IN_REPORT = "facility_not_in_report"
REASON_OUTSIDE_WINDOW = "outside_window"


# ---------------------------------------------------------------------------
# Proof sheets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptedEntry:
    note_id: str
    strip_id: str
    asset_id: str
    physical: Fraction
    anchor: Fraction

    def to_canonical(self) -> dict:
        return {
            "note_id": self.note_id,
            "strip_id": self.strip_id,
            "asset_id": self.asset_id,
            "physical": self.physical,
            "anchor": self.anchor,
        }


@dataclass(frozen=True)
class RejectedEntry:
    note_id: str
    strip_id: str
    asset_id: str
    reason: str
    detail: str

    def to_canonical(self) -> dict:
        return {
            "note_id": self.note_id,
            "strip_id": self.strip_id,
            "asset_id": self.asset_id,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ProofSheet:
    sheet_id: str
    report_id: str
    panel_id: str
    panel_revision: str
    accepted: tuple[AcceptedEntry, ...]
    rejected: tuple[RejectedEntry, ...]

    def canonical_content(self) -> dict:
        return {
            "report_id": self.report_id,
            "panel_id": self.panel_id,
            "panel_revision": self.panel_revision,
            "accepted": [e.to_canonical() for e in self.accepted],
            "rejected": [e.to_canonical() for e in self.rejected],
        }

    def fingerprint(self) -> str:
        return fingerprint_of(self.canonical_content())

    def to_dict(self) -> dict:
        data = canonical(self.canonical_content())
        data["sheet_id"] = self.sheet_id
        data["fingerprint"] = self.fingerprint()
        return data


# ---------------------------------------------------------------------------
# Line issues and plates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineIssue:
    """One typesetting of a line spec under one panel revision."""

    line_id: str
    issue_seq: int
    sheet_id: str
    panel_revision: str
    spec: dict  # canonical spec (exact 'n/d' text)
    anchors: dict  # note_id -> Fraction (the anchors actually used)
    solution: dict  # bench output (Fractions in boxes)

    def fingerprint(self) -> str:
        return fingerprint_of(
            {
                "line_id": self.line_id,
                "sheet_id": self.sheet_id,
                "panel_revision": self.panel_revision,
                "spec": self.spec,
                "anchors": self.anchors,
                "solution": self.solution,
            }
        )


@dataclass(frozen=True)
class Plate:
    """A frozen daily batch of line issues."""

    plate_id: str
    batch: str
    lines: tuple[tuple[str, int], ...]  # (line_id, issue_seq), sorted

    def fingerprint(self, issue_fingerprints: dict[tuple[str, int], str]) -> str:
        return fingerprint_of(
            {
                "batch": self.batch,
                "lines": [
                    [line_id, seq, issue_fingerprints[(line_id, seq)]]
                    for line_id, seq in self.lines
                ],
            }
        )


# ---------------------------------------------------------------------------
# Line spec canonicalisation
# ---------------------------------------------------------------------------


def canonical_line_spec(spec_dict: dict) -> dict:
    """Normalise a line spec dict into its canonical (identity) form.

    Items are sorted by note_id and forbidden zones are sorted, so the
    same line written in a different order is the same line.  Placement
    order is preserved: the placement index is the 贴法号.
    """
    items = []
    for it in spec_dict["items"]:
        items.append(
            {
                "note_id": str(it["note_id"]),
                "required": bool(it.get("required", False)),
                "value": int(it.get("value", 0)),
                "placements": [
                    [fraction_to_db(as_fraction(lo)), fraction_to_db(as_fraction(hi))]
                    for lo, hi in it["placements"]
                ],
            }
        )
    items.sort(key=lambda d: d["note_id"])
    forbidden = sorted(
        [fraction_to_db(as_fraction(a)), fraction_to_db(as_fraction(b))]
        for a, b in spec_dict.get("forbidden", [])
    )
    return {
        "line_id": str(spec_dict["line_id"]),
        "sheet_id": str(spec_dict["sheet_id"]),
        "width": fraction_to_db(as_fraction(spec_dict["width"])),
        "forbidden": forbidden,
        "items": items,
    }


def line_spec_from_canonical(
    canonical_spec: dict, anchors: dict[str, Fraction]
) -> LineSpec:
    """Build a bench LineSpec from a canonical spec and an anchor table."""
    items = []
    for it in canonical_spec["items"]:
        note_id = it["note_id"]
        if note_id not in anchors:
            raise LineSpecError(
                f"note {note_id!r} is not an accepted entry of sheet "
                f"{canonical_spec['sheet_id']!r}"
            )
        items.append(
            LineItem(
                note_id=note_id,
                anchor=anchors[note_id],
                required=it["required"],
                value=it["value"],
                placements=tuple(
                    Placement(as_fraction(lo), as_fraction(hi))
                    for lo, hi in it["placements"]
                ),
            )
        )
    return LineSpec(
        line_id=canonical_spec["line_id"],
        width=as_fraction(canonical_spec["width"]),
        forbidden=tuple(
            (as_fraction(a), as_fraction(b))
            for a, b in canonical_spec["forbidden"]
        ),
        items=tuple(items),
    )


# ---------------------------------------------------------------------------
# The press room
# ---------------------------------------------------------------------------


class PressRoom:
    def __init__(self, lib: InspectionLibrary, panels: PanelCatalog):
        self.lib = lib
        self.panels = panels
        self._sheets: dict[str, ProofSheet] = {}
        self._lines: dict[str, list[LineIssue]] = {}
        self._plates: dict[str, Plate] = {}

    # ------------------------------------------------------------------
    # Annotation sheets
    # ------------------------------------------------------------------
    def issue_sheet(self, spec: dict) -> ProofSheet:
        """Issue an annotation sheet from a report and a panel revision."""
        sheet_id = str(spec["sheet_id"])
        if not sheet_id:
            raise SheetError("sheet_id must be non-empty")
        report_id = str(spec["report_id"])
        try:
            report = self.lib.get_report(report_id)
        except KeyError as exc:
            raise SheetError(
                f"sheet {sheet_id!r} names unknown report {report_id!r}; "
                f"the sheet is not issued"
            ) from exc
        panel_id = str(spec["panel_id"])
        panel_revision = spec.get("panel_revision")
        try:
            prev = self.panels.get(
                panel_id, None if panel_revision is None else str(panel_revision)
            )
        except PanelError as exc:
            raise SheetError(f"sheet {sheet_id!r} not issued: {exc}") from exc
        entries = spec.get("entries") or []
        if not entries:
            raise SheetError(f"sheet {sheet_id!r} has no entries")
        sheet = self._build_proof(sheet_id, report, prev, entries)
        existing = self._sheets.get(sheet_id)
        if existing is not None:
            if existing.fingerprint() != sheet.fingerprint():
                raise SheetConflictError(
                    f"sheet_id {sheet_id!r} already exists with different "
                    f"content"
                )
            return existing
        self._sheets[sheet_id] = sheet
        return sheet

    def _build_proof(self, sheet_id, report, prev, entries) -> ProofSheet:
        facilities = {f.asset_id: f.physical_position for f in report.facilities}
        accepted: list[AcceptedEntry] = []
        rejected: list[RejectedEntry] = []
        seen = set()
        for e in entries:
            note_id = str(e["note_id"])
            strip_id = str(e["strip_id"])
            asset_id = str(e["asset_id"])
            if note_id in seen:
                raise SheetError(
                    f"sheet {sheet_id!r} has duplicate note_id {note_id!r}"
                )
            seen.add(note_id)
            strip = prev.strip_or_none(strip_id)
            if strip is None:
                rejected.append(
                    RejectedEntry(
                        note_id, strip_id, asset_id, REASON_UNKNOWN_STRIP,
                        f"panel {prev.panel_id}/{prev.panel_revision} has no "
                        f"strip {strip_id!r}",
                    )
                )
                continue
            if strip.catalog_revision != report.catalog_revision:
                rejected.append(
                    RejectedEntry(
                        note_id, strip_id, asset_id, REASON_REVISION_MISMATCH,
                        f"strip {strip_id!r} is bound to catalog revision "
                        f"{strip.catalog_revision}, but report "
                        f"{report.report_id} was resolved under "
                        f"{report.catalog_revision}",
                    )
                )
                continue
            physical = facilities.get(asset_id)
            if physical is None:
                rejected.append(
                    RejectedEntry(
                        note_id, strip_id, asset_id,
                        REASON_FACILITY_NOT_IN_REPORT,
                        f"report {report.report_id} does not cover facility "
                        f"{asset_id!r}",
                    )
                )
                continue
            try:
                delivered = self.panels.anchor(
                    prev.panel_id, strip_id, physical, prev.panel_revision
                )
            except OutsideWindowError as exc:
                rejected.append(
                    RejectedEntry(
                        note_id, strip_id, asset_id, REASON_OUTSIDE_WINDOW,
                        str(exc),
                    )
                )
                continue
            accepted.append(
                AcceptedEntry(
                    note_id, strip_id, asset_id, physical, delivered["anchor"]
                )
            )
        accepted.sort(key=lambda e: e.note_id)
        rejected.sort(key=lambda e: e.note_id)
        return ProofSheet(
            sheet_id=sheet_id,
            report_id=report.report_id,
            panel_id=prev.panel_id,
            panel_revision=prev.panel_revision,
            accepted=tuple(accepted),
            rejected=tuple(rejected),
        )

    def get_sheet(self, sheet_id: str) -> ProofSheet:
        try:
            return self._sheets[sheet_id]
        except KeyError as exc:
            raise SheetError(f"no sheet {sheet_id!r}") from exc

    def sheet_view(self, sheet_id: str) -> dict:
        """Render a sheet with each anchor's projection basis attached."""
        sheet = self.get_sheet(sheet_id)
        data = sheet.to_dict()
        for entry in data["accepted"]:
            delivered = self.panels.anchor(
                sheet.panel_id,
                entry["strip_id"],
                as_fraction(entry["physical"]),
                sheet.panel_revision,
            )
            entry["basis"] = canonical(delivered["basis"])
        return data

    def verify_sheet(self, sheet_id: str) -> dict:
        """Offline re-check: rebuild the proof from retained revisions.

        Everything needed -- the report, the bound catalog revision and the
        panel revision -- is retained in the store, so an old sheet's proof
        can be re-derived and the fingerprints compared without any
        outside input.
        """
        sheet = self.get_sheet(sheet_id)
        report = self.lib.get_report(sheet.report_id)
        prev = self.panels.get(sheet.panel_id, sheet.panel_revision)
        entries = [
            {"note_id": e.note_id, "strip_id": e.strip_id, "asset_id": e.asset_id}
            for e in (*sheet.accepted, *sheet.rejected)
        ]
        rebuilt = self._build_proof(sheet.sheet_id, report, prev, entries)
        return {
            "sheet_id": sheet.sheet_id,
            "verified": rebuilt.fingerprint() == sheet.fingerprint(),
            "fingerprint": sheet.fingerprint(),
            "anchors_checked": len(sheet.accepted),
            "report_fingerprint": report.fingerprint(),
            "panel_revision": sheet.panel_revision,
            "catalog_revision": report.catalog_revision,
        }

    # ------------------------------------------------------------------
    # Anchors for the bench
    # ------------------------------------------------------------------
    def anchor_table(
        self, sheet_id: str, panel_revision: Optional[str] = None
    ) -> dict[str, Fraction]:
        """Anchors of a sheet's accepted entries under a panel revision."""
        sheet = self.get_sheet(sheet_id)
        out = {}
        for e in sheet.accepted:
            delivered = self.panels.anchor(
                sheet.panel_id, e.strip_id, e.physical,
                panel_revision or sheet.panel_revision,
            )
            out[e.note_id] = delivered["anchor"]
        return out

    # ------------------------------------------------------------------
    # Typesetting (新排版 -- public operation)
    # ------------------------------------------------------------------
    def typeset_line(self, spec_dict: dict) -> LineIssue:
        """Typeset a line against its sheet's own panel revision."""
        canonical_spec = canonical_line_spec(spec_dict)
        sheet_id = canonical_spec["sheet_id"]
        try:
            sheet = self._sheets[sheet_id]
        except KeyError as exc:
            raise LineSpecError(
                f"line {canonical_spec['line_id']!r} names unknown sheet "
                f"{sheet_id!r}"
            ) from exc
        anchors = {e.note_id: e.anchor for e in sheet.accepted}
        return self._typeset(canonical_spec, sheet, sheet.panel_revision, anchors)

    def _typeset(
        self,
        canonical_spec: dict,
        sheet: ProofSheet,
        panel_revision: str,
        anchors: dict[str, Fraction],
    ) -> LineIssue:
        spec = line_spec_from_canonical(canonical_spec, anchors)
        solution = bench_typeset(spec)
        line_id = canonical_spec["line_id"]
        issues = self._lines.setdefault(line_id, [])
        if issues and issues[0].spec != canonical_spec:
            raise LineConflictError(
                f"line {line_id!r} already exists with a different spec; "
                f"use a new line_id for a different line"
            )
        used = {it.note_id: anchors[it.note_id] for it in spec.items}
        issue = LineIssue(
            line_id=line_id,
            issue_seq=len(issues) + 1,
            sheet_id=sheet.sheet_id,
            panel_revision=panel_revision,
            spec=canonical_spec,
            anchors=used,
            solution=solution,
        )
        if issues and issues[-1].fingerprint() == issue.fingerprint():
            return issues[-1]  # identical re-typeset is idempotent
        issues.append(issue)
        return issue

    def get_issue(self, line_id: str, issue_seq: Optional[int] = None) -> LineIssue:
        issues = self._lines.get(line_id)
        if not issues:
            raise PlateError(f"no line {line_id!r}")
        if issue_seq is None:
            return issues[-1]
        for issue in issues:
            if issue.issue_seq == issue_seq:
                return issue
        raise PlateError(f"no issue {issue_seq} of line {line_id!r}")

    # ------------------------------------------------------------------
    # Plates (daily batch freeze) and re-issue
    # ------------------------------------------------------------------
    def freeze_plate(
        self,
        plate_id: str,
        batch: str,
        refs: Sequence,
    ) -> Plate:
        """Freeze a batch of lines.  ``refs`` are line_ids (latest issue)
        or ``(line_id, issue_seq)`` pairs."""
        if not plate_id:
            raise PlateError("plate_id must be non-empty")
        if not refs:
            raise PlateError(f"plate {plate_id!r} freezes no lines")
        lines = []
        for ref in refs:
            if isinstance(ref, (list, tuple)):
                line_id, seq = str(ref[0]), int(ref[1])
            else:
                line_id, seq = str(ref), None
            issue = self.get_issue(line_id, seq)
            lines.append((issue.line_id, issue.issue_seq))
        lines.sort()
        plate = Plate(plate_id, str(batch), tuple(lines))
        existing = self._plates.get(plate_id)
        if existing is not None:
            if existing != plate:
                raise PlateConflictError(
                    f"plate_id {plate_id!r} already frozen with different "
                    f"content"
                )
            return existing
        self._plates[plate_id] = plate
        return plate

    def plate_fingerprint(self, plate_id: str) -> str:
        plate = self._plates[plate_id]
        fps = {
            (line_id, seq): self.get_issue(line_id, seq).fingerprint()
            for line_id, seq in plate.lines
        }
        return plate.fingerprint(fps)

    def reissue_plate(
        self,
        plate_id: str,
        panel_revision: str,
        new_plate_id: str,
        batch: str,
    ) -> dict:
        """Re-issue a frozen plate after a paper change.

        Only lines whose anchors actually moved under ``panel_revision``
        are re-typeset; the rest carry their frozen issue into the new
        plate unchanged.
        """
        try:
            plate = self._plates[plate_id]
        except KeyError as exc:
            raise PlateError(f"no plate {plate_id!r}") from exc
        affected: list[str] = []
        unaffected: list[str] = []
        new_refs = []
        new_issues: list[LineIssue] = []
        for line_id, seq in plate.lines:
            issue = self.get_issue(line_id, seq)
            sheet = self.get_sheet(issue.sheet_id)
            # The target revision must exist for this sheet's panel.
            self.panels.get(sheet.panel_id, panel_revision)
            fresh = self.anchor_table(sheet.sheet_id, panel_revision)
            used = {nid: fresh[nid] for nid in issue.anchors}
            if used == issue.anchors:
                unaffected.append(line_id)
                new_refs.append((line_id, seq))
                continue
            new_issue = self._typeset(issue.spec, sheet, panel_revision, used)
            affected.append(line_id)
            new_issues.append(new_issue)
            new_refs.append((new_issue.line_id, new_issue.issue_seq))
        new_plate = self.freeze_plate(new_plate_id, batch, new_refs)
        return {
            "plate_id": new_plate.plate_id,
            "batch": new_plate.batch,
            "panel_revision": panel_revision,
            "affected": affected,
            "unaffected": unaffected,
            "new_issues": new_issues,
        }


# ---------------------------------------------------------------------------
# SQLite persistence (its own schema, like the other two)
# ---------------------------------------------------------------------------


PRESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sheet (
    sheet_id       TEXT PRIMARY KEY,
    report_id      TEXT NOT NULL,
    panel_id       TEXT NOT NULL,
    panel_revision TEXT NOT NULL,
    fingerprint    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sheet_accept (
    sheet_id   TEXT NOT NULL REFERENCES sheet(sheet_id),
    note_id    TEXT NOT NULL,
    strip_id   TEXT NOT NULL,
    asset_id   TEXT NOT NULL,
    phys_num   TEXT NOT NULL, phys_den TEXT NOT NULL,
    anchor_num TEXT NOT NULL, anchor_den TEXT NOT NULL,
    PRIMARY KEY (sheet_id, note_id)
);
CREATE TABLE IF NOT EXISTS sheet_reject (
    sheet_id TEXT NOT NULL REFERENCES sheet(sheet_id),
    note_id  TEXT NOT NULL,
    strip_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    reason   TEXT NOT NULL,
    detail   TEXT NOT NULL,
    PRIMARY KEY (sheet_id, note_id)
);
CREATE TABLE IF NOT EXISTS line_issue (
    line_id        TEXT NOT NULL,
    issue_seq      INTEGER NOT NULL,
    sheet_id       TEXT NOT NULL,
    panel_revision TEXT NOT NULL,
    spec_json      TEXT NOT NULL,
    anchors_json   TEXT NOT NULL,
    solution_json  TEXT NOT NULL,
    fingerprint    TEXT NOT NULL,
    PRIMARY KEY (line_id, issue_seq)
);
CREATE TABLE IF NOT EXISTS plate (
    plate_id TEXT PRIMARY KEY,
    batch    TEXT NOT NULL,
    lines_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL
);
"""


def ensure_press_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(PRESS_SCHEMA)


def persist_sheet(conn: sqlite3.Connection, sheet: ProofSheet) -> None:
    with conn:
        row = conn.execute(
            "SELECT fingerprint FROM sheet WHERE sheet_id = ?", (sheet.sheet_id,)
        ).fetchone()
        if row is not None:
            if row["fingerprint"] != sheet.fingerprint():
                raise SheetConflictError(
                    f"sheet_id {sheet.sheet_id!r} stored with different content"
                )
            return
        conn.execute(
            "INSERT INTO sheet (sheet_id, report_id, panel_id, panel_revision, "
            "fingerprint) VALUES (?, ?, ?, ?, ?)",
            (
                sheet.sheet_id, sheet.report_id, sheet.panel_id,
                sheet.panel_revision, sheet.fingerprint(),
            ),
        )
        for e in sheet.accepted:
            conn.execute(
                "INSERT INTO sheet_accept (sheet_id, note_id, strip_id, "
                "asset_id, phys_num, phys_den, anchor_num, anchor_den) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sheet.sheet_id, e.note_id, e.strip_id, e.asset_id,
                    str(e.physical.numerator), str(e.physical.denominator),
                    str(e.anchor.numerator), str(e.anchor.denominator),
                ),
            )
        for e in sheet.rejected:
            conn.execute(
                "INSERT INTO sheet_reject (sheet_id, note_id, strip_id, "
                "asset_id, reason, detail) VALUES (?, ?, ?, ?, ?, ?)",
                (sheet.sheet_id, e.note_id, e.strip_id, e.asset_id,
                 e.reason, e.detail),
            )


def load_sheets(conn: sqlite3.Connection) -> dict[str, ProofSheet]:
    out = {}
    for row in conn.execute("SELECT * FROM sheet ORDER BY sheet_id").fetchall():
        sid = row["sheet_id"]
        accepted = tuple(
            AcceptedEntry(
                note_id=r["note_id"],
                strip_id=r["strip_id"],
                asset_id=r["asset_id"],
                physical=Fraction(int(r["phys_num"]), int(r["phys_den"])),
                anchor=Fraction(int(r["anchor_num"]), int(r["anchor_den"])),
            )
            for r in conn.execute(
                "SELECT * FROM sheet_accept WHERE sheet_id = ? ORDER BY note_id",
                (sid,),
            ).fetchall()
        )
        rejected = tuple(
            RejectedEntry(
                note_id=r["note_id"],
                strip_id=r["strip_id"],
                asset_id=r["asset_id"],
                reason=r["reason"],
                detail=r["detail"],
            )
            for r in conn.execute(
                "SELECT * FROM sheet_reject WHERE sheet_id = ? ORDER BY note_id",
                (sid,),
            ).fetchall()
        )
        out[sid] = ProofSheet(
            sheet_id=sid,
            report_id=row["report_id"],
            panel_id=row["panel_id"],
            panel_revision=row["panel_revision"],
            accepted=accepted,
            rejected=rejected,
        )
    return out


def persist_line_issue(conn: sqlite3.Connection, issue: LineIssue) -> None:
    import json as _json

    with conn:
        row = conn.execute(
            "SELECT fingerprint FROM line_issue WHERE line_id = ? AND "
            "issue_seq = ?",
            (issue.line_id, issue.issue_seq),
        ).fetchone()
        if row is not None:
            if row["fingerprint"] != issue.fingerprint():
                raise LineConflictError(
                    f"line {issue.line_id!r} issue {issue.issue_seq} stored "
                    f"with different content"
                )
            return
        conn.execute(
            "INSERT INTO line_issue (line_id, issue_seq, sheet_id, "
            "panel_revision, spec_json, anchors_json, solution_json, "
            "fingerprint) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                issue.line_id, issue.issue_seq, issue.sheet_id,
                issue.panel_revision,
                _json.dumps(canonical(issue.spec), sort_keys=True),
                _json.dumps(canonical(issue.anchors), sort_keys=True),
                _json.dumps(canonical(issue.solution), sort_keys=True),
                issue.fingerprint(),
            ),
        )


def load_line_issues(conn: sqlite3.Connection) -> dict[str, list[LineIssue]]:
    import json as _json

    out: dict[str, list[LineIssue]] = {}
    rows = conn.execute(
        "SELECT * FROM line_issue ORDER BY line_id, issue_seq"
    ).fetchall()
    for r in rows:
        anchors = {
            k: Fraction(v)
            for k, v in _json.loads(r["anchors_json"]).items()
        }
        issue = LineIssue(
            line_id=r["line_id"],
            issue_seq=r["issue_seq"],
            sheet_id=r["sheet_id"],
            panel_revision=r["panel_revision"],
            spec=_json.loads(r["spec_json"]),
            anchors=anchors,
            solution=_json.loads(r["solution_json"]),
        )
        out.setdefault(issue.line_id, []).append(issue)
    return out


def persist_plate(conn: sqlite3.Connection, room: PressRoom, plate: Plate) -> None:
    import json as _json

    fp = room.plate_fingerprint(plate.plate_id)
    with conn:
        row = conn.execute(
            "SELECT fingerprint FROM plate WHERE plate_id = ?", (plate.plate_id,)
        ).fetchone()
        if row is not None:
            if row["fingerprint"] != fp:
                raise PlateConflictError(
                    f"plate_id {plate.plate_id!r} stored with different content"
                )
            return
        conn.execute(
            "INSERT INTO plate (plate_id, batch, lines_json, fingerprint) "
            "VALUES (?, ?, ?, ?)",
            (
                plate.plate_id, plate.batch,
                _json.dumps([list(t) for t in plate.lines]), fp,
            ),
        )


def load_plates(conn: sqlite3.Connection) -> dict[str, Plate]:
    import json as _json

    out = {}
    for r in conn.execute("SELECT * FROM plate ORDER BY plate_id").fetchall():
        lines = tuple((line_id, int(seq)) for line_id, seq in
                      _json.loads(r["lines_json"]))
        out[r["plate_id"]] = Plate(r["plate_id"], r["batch"], lines)
    return out
