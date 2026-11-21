"""Typesetting bench: exact 1D layout for one print line.

Pure layout mathematics -- this module knows nothing about routes, panels
or reports.  A **line** has a width (paper x in ``[0, width)``), zero or
more half-open forbidden zones (禁印区), and up to 18 items (地名).  Each
item carries an anchor on the line and 1..3 allowed **placements** (贴法);
placement ``i`` is the half-open glyph box ``[anchor + lo_i, anchor +
hi_i)``.

Rules
-----
* Every **required** item (必印项) must be printed with exactly one of its
  placements.
* Every **optional** item carries a positive integer value and may be
  printed at most once (one placement) or not at all.
* A glyph box must lie inside ``[0, width]`` and must not intersect any
  forbidden zone or any other chosen box.  Boxes are half-open, so boxes
  that merely *touch* (one ends exactly where the next begins) are fine.
* Objective: maximise the total value of printed optional items.
* Ties in value are broken by the lexicographically smallest ascending
  list of ``(note_id, placement_index)`` pairs of *all* printed items
  (required items are always printed, so their chosen placement index is
  part of the list).

Infeasibility is reported with a concrete cause: either a required item
has no placement that fits the paper conditions at all, or a specific
(irreducible) set of required items cannot be placed simultaneously.

Algorithm: exact depth-first branch and bound over items (required first,
then optional by descending value), pruning with the suffix sum of
remaining optional values.  Worst case is exponential -- the problem is a
multiple-choice weighted interval scheduling -- but the stated limits
(<= 18 items, <= 3 placements each) keep the search small, and equal-value
branches are still explored so the lexicographic tie-break stays exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

from chainage_catalog import Interval, as_fraction, merge_intervals

MAX_ITEMS = 18
MAX_PLACEMENTS = 3


class LineSpecError(ValueError):
    """A line spec violates the bench's structural limits."""


@dataclass(frozen=True)
class Placement:
    """Half-open glyph box relative to the anchor: [anchor+lo, anchor+hi)."""

    offset_start: Fraction
    offset_end: Fraction


@dataclass(frozen=True)
class LineItem:
    note_id: str
    anchor: Fraction
    required: bool
    value: int  # 0 for required items; positive integer for optional ones
    placements: tuple[Placement, ...]


@dataclass(frozen=True)
class LineSpec:
    line_id: str
    width: Fraction
    forbidden: tuple[tuple[Fraction, Fraction], ...]  # half-open zones
    items: tuple[LineItem, ...]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(spec: LineSpec) -> None:
    if spec.width <= 0:
        raise LineSpecError("line width must be positive")
    if not 1 <= len(spec.items) <= MAX_ITEMS:
        raise LineSpecError(
            f"line {spec.line_id!r} allows 1..{MAX_ITEMS} items, "
            f"got {len(spec.items)}"
        )
    ids = [it.note_id for it in spec.items]
    if len(set(ids)) != len(ids):
        raise LineSpecError(f"line {spec.line_id!r} has duplicate note_ids")
    for it in spec.items:
        if not 1 <= len(it.placements) <= MAX_PLACEMENTS:
            raise LineSpecError(
                f"item {it.note_id!r} allows 1..{MAX_PLACEMENTS} placements, "
                f"got {len(it.placements)}"
            )
        for pl in it.placements:
            if not pl.offset_start < pl.offset_end:
                raise LineSpecError(
                    f"item {it.note_id!r} has a non-positive placement "
                    f"[{pl.offset_start}, {pl.offset_end})"
                )
        if it.required:
            if it.value != 0:
                raise LineSpecError(
                    f"required item {it.note_id!r} must carry value 0"
                )
        else:
            if (
                isinstance(it.value, bool)
                or not isinstance(it.value, int)
                or it.value <= 0
            ):
                raise LineSpecError(
                    f"optional item {it.note_id!r} needs a positive integer "
                    f"value, got {it.value!r}"
                )
    for z in spec.forbidden:
        if not z[0] < z[1]:
            raise LineSpecError(
                f"forbidden zone [{z[0]}, {z[1]}) is not positive length"
            )


# ---------------------------------------------------------------------------
# Geometry helpers (half-open boxes; touching is allowed)
# ---------------------------------------------------------------------------


def _overlap(s1: Fraction, e1: Fraction, s2: Fraction, e2: Fraction) -> bool:
    return s1 < e2 and s2 < e1


def _feasible_placements(
    item: LineItem, width: Fraction, zones: list[Interval]
) -> list[tuple[int, Fraction, Fraction]]:
    """(placement_index, box_start, box_end) that fit width and zones.

    Placements producing an identical box are collapsed to the smallest
    index: any solution using the larger index could use the smaller one
    with the same value and a lexicographically smaller tie list, so the
    dropped duplicates can never belong to the returned optimum.
    """
    out = []
    seen_boxes = set()
    for idx, pl in enumerate(item.placements):
        s = item.anchor + pl.offset_start
        e = item.anchor + pl.offset_end
        if s < 0 or e > width:
            continue
        if any(_overlap(s, e, z.start, z.end) for z in zones):
            continue
        if (s, e) in seen_boxes:
            continue
        seen_boxes.add((s, e))
        out.append((idx, s, e))
    return out


def _any_assignment(
    positions: Sequence[int], feas: list[list[tuple[int, Fraction, Fraction]]]
) -> bool:
    """True if the given items can be placed simultaneously (any choice)."""
    placed: list[tuple[Fraction, Fraction]] = []

    def dfs(k: int) -> bool:
        if k == len(positions):
            return True
        for _, s, e in feas[positions[k]]:
            if all(not _overlap(s, e, ps, pe) for ps, pe in placed):
                placed.append((s, e))
                if dfs(k + 1):
                    return True
                placed.pop()
        return False

    return dfs(0)


def _irreducible_infeasible(
    positions: Sequence[int], feas: list[list[tuple[int, Fraction, Fraction]]]
) -> list[int]:
    """A minimal (under removal) subset of items that is still infeasible."""
    culprit = list(positions)
    for pos in list(culprit):
        trial = [p for p in culprit if p != pos]
        if trial and not _any_assignment(trial, feas):
            culprit = trial
    return culprit


# ---------------------------------------------------------------------------
# The bench
# ---------------------------------------------------------------------------


def typeset(spec: LineSpec) -> dict:
    """Solve one line exactly.

    Returns either ``{"status": "ok", "value", "assignment", "tie_list"}``
    or ``{"status": "infeasible", "reason": {...}}``.  ``assignment`` is a
    list of ``{"note_id", "placement", "box": (start, end)}`` sorted by
    note_id; ``tie_list`` is the ascending (note_id, placement) list used
    for the tie-break.
    """
    _validate(spec)
    zones = merge_intervals(Interval(s, e) for s, e in spec.forbidden)

    feas = [_feasible_placements(it, spec.width, zones) for it in spec.items]

    # A required item that cannot be placed at all: paper conditions.
    for it, opts in zip(spec.items, feas):
        if it.required and not opts:
            return {
                "status": "infeasible",
                "reason": {
                    "cause": "no_feasible_placement",
                    "item": it.note_id,
                    "detail": (
                        f"required item {it.note_id!r} has no placement "
                        f"inside line width and outside the forbidden zones"
                    ),
                },
            }

    # Required items that only fail together: mutual conflict.
    req = [i for i, it in enumerate(spec.items) if it.required]
    if not _any_assignment(req, feas):
        subset = _irreducible_infeasible(req, feas)
        ids = sorted(spec.items[i].note_id for i in subset)
        return {
            "status": "infeasible",
            "reason": {
                "cause": "required_items_conflict",
                "items": ids,
                "detail": (
                    f"required items {ids} cannot be placed simultaneously "
                    f"without their boxes intersecting"
                ),
            },
        }

    # Exact maximisation over optional values with lexicographic tie-break.
    order = sorted(
        range(len(spec.items)),
        key=lambda i: (
            not spec.items[i].required,  # required first
            -spec.items[i].value,  # then high value first (tighter bounds)
            spec.items[i].note_id,
        ),
    )
    suffix = [0] * (len(order) + 1)
    for k in range(len(order) - 1, -1, -1):
        suffix[k] = suffix[k + 1] + spec.items[order[k]].value

    best_value = -1
    best_list: Optional[list] = None
    best_sol: list = []
    placed: list[tuple[int, int, Fraction, Fraction]] = []

    def dfs(k: int, value: int) -> None:
        nonlocal best_value, best_list, best_sol
        if value + suffix[k] < best_value:
            return  # cannot reach the best value; equal-value ties explored
        if k == len(order):
            tl = sorted(
                (spec.items[pos].note_id, pidx) for pos, pidx, _, _ in placed
            )
            if value > best_value or best_list is None or tl < best_list:
                best_value = value
                best_list = tl
                best_sol = list(placed)
            return
        pos = order[k]
        it = spec.items[pos]
        for pidx, s, e in feas[pos]:
            if all(not _overlap(s, e, ps, pe) for _, _, ps, pe in placed):
                placed.append((pos, pidx, s, e))
                dfs(k + 1, value + it.value)
                placed.pop()
        if not it.required:
            dfs(k + 1, value)  # skip: optional items appear at most once

    dfs(0, 0)

    assignment = sorted(
        (
            {
                "note_id": spec.items[pos].note_id,
                "placement": pidx,
                "box": (s, e),
            }
            for pos, pidx, s, e in best_sol
        ),
        key=lambda a: a["note_id"],
    )
    return {
        "status": "ok",
        "value": best_value,
        "assignment": assignment,
        "tie_list": [list(pair) for pair in best_list],
    }
