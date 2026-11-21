"""Tests for the typesetting bench (exact line layout).

Run:  python3 -m unittest discover -s solution/tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typeset_bench import (  # noqa: E402
    LineItem,
    LineSpec,
    LineSpecError,
    Placement,
    typeset,
)


def item(note_id, anchor, required=False, value=0, placements=((0, 3),)):
    return LineItem(
        note_id=note_id,
        anchor=Fraction(anchor),
        required=required,
        value=value,
        placements=tuple(
            Placement(Fraction(lo), Fraction(hi)) for lo, hi in placements
        ),
    )


def spec(items, width=10, forbidden=(), line_id="L"):
    return LineSpec(
        line_id=line_id,
        width=Fraction(width),
        forbidden=tuple((Fraction(a), Fraction(b)) for a, b in forbidden),
        items=tuple(items),
    )


def boxes(solution):
    return {a["note_id"]: a["box"] for a in solution["assignment"]}


class TestWorkedExample(unittest.TestCase):
    """The exact scenario from the task statement."""

    def test_touching_boxes_both_print(self):
        # P@2 required, Q@5 value 4, placements [anchor, anchor+3).
        sol = typeset(spec([
            item("P", 2, required=True),
            item("Q", 5, value=4),
        ]))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 4)
        self.assertEqual(
            boxes(sol),
            {"P": (Fraction(2), Fraction(5)), "Q": (Fraction(5), Fraction(8))},
        )

    def test_forbidden_zone_drops_optional(self):
        sol = typeset(spec(
            [item("P", 2, required=True), item("Q", 5, value=4)],
            forbidden=[(7, 8)],
        ))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 0)
        self.assertEqual(set(boxes(sol)), {"P"})

    def test_paper_end_8_overlapping_anchors(self):
        # After the paper change the anchors are 8/5 and 4; the boxes
        # [8/5, 23/5) and [4, 7) overlap, so only the required P prints.
        p = Fraction(8, 5)
        sol = typeset(spec([
            item("P", p, required=True),
            item("Q", 4, value=4),
        ]))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 0)
        self.assertEqual(
            boxes(sol), {"P": (Fraction(8, 5), Fraction(8, 5) + 3)}
        )


class TestGeometry(unittest.TestCase):
    def test_touching_forbidden_zone_is_allowed(self):
        # Box [2,7) touches zone [7,8): legal.
        sol = typeset(spec(
            [item("P", 2, required=True, placements=((0, 5),))],
            forbidden=[(7, 8)],
        ))
        self.assertEqual(sol["status"], "ok")

    def test_box_may_end_exactly_at_line_width(self):
        sol = typeset(spec(
            [item("P", 7, required=True, placements=((0, 3),))],
        ))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(boxes(sol)["P"], (Fraction(7), Fraction(10)))

    def test_box_past_width_rejected(self):
        sol = typeset(spec(
            [item("P", 8, required=True, placements=((0, 3),))],
        ))
        self.assertEqual(sol["status"], "infeasible")
        self.assertEqual(sol["reason"]["cause"], "no_feasible_placement")
        self.assertEqual(sol["reason"]["item"], "P")

    def test_negative_box_rejected(self):
        sol = typeset(spec(
            [item("P", 1, required=True, placements=((-2, 1),))],
        ))
        self.assertEqual(sol["status"], "infeasible")
        self.assertEqual(sol["reason"]["cause"], "no_feasible_placement")

    def test_reverse_anchors(self):
        # 逆排 strip: P at anchor 8, Q at anchor 5, boxes behind the anchor.
        sol = typeset(spec([
            item("P", 8, required=True, placements=((-3, 0),)),
            item("Q", 5, value=4, placements=((-3, 0),)),
        ]))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 4)
        self.assertEqual(
            boxes(sol),
            {"P": (Fraction(5), Fraction(8)), "Q": (Fraction(2), Fraction(5))},
        )


class TestOptimisation(unittest.TestCase):
    def test_higher_value_wins_exclusive_slot(self):
        sol = typeset(spec([
            item("A", 2, value=3),
            item("B", 2, value=9),  # same box, only one fits
        ]))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 9)
        self.assertEqual(set(boxes(sol)), {"B"})

    def test_optional_appears_at_most_once(self):
        # Two non-overlapping placements for one item: only one is used.
        sol = typeset(spec([
            item("A", 2, value=5, placements=((0, 2), (4, 6))),
        ]))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 5)
        self.assertEqual(len(sol["assignment"]), 1)

    def test_placement_choice_unblocks_value(self):
        # Required A must pick the placement that lets optional B print.
        sol = typeset(spec([
            item("A", 2, required=True, placements=((0, 3), (4, 6))),
            item("B", 3, value=7, placements=((0, 2),)),
        ]))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 7)
        self.assertEqual(boxes(sol)["A"], (Fraction(6), Fraction(8)))
        self.assertEqual(boxes(sol)["B"], (Fraction(3), Fraction(5)))

    def test_tie_break_lexicographic(self):
        # Equal values, mutually exclusive: the lexicographically smaller
        # ascending (note_id, placement) list wins -> [(N3,0)] < [(N4,0)].
        sol = typeset(spec([
            item("N4", 5, value=4, placements=((0, 3),)),
            item("N3", 8, value=4, placements=((-3, 0),)),
        ]))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 4)
        self.assertEqual(set(boxes(sol)), {"N3"})
        self.assertEqual(sol["tie_list"], [["N3", 0]])

    def test_tie_break_prefers_smaller_placement_index(self):
        # Same items printed either way; placement 0 of B is lex-smaller.
        sol = typeset(spec([
            item("A", 2, required=True),
            item("B", 5, value=4, placements=((0, 3), (0, 2))),
        ]))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["tie_list"], [["A", 0], ["B", 0]])

    def test_many_items_exact(self):
        # 18 items, 3 placements each: a chain of touching unit boxes.
        items = [
            item(f"N{i:02d}", i, required=(i % 2 == 0), value=1 if i % 2 else 0,
                 placements=((0, 1), (0, 1), (0, 1)))
            for i in range(18)
        ]
        sol = typeset(spec(items, width=18))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 9)


class TestInfeasible(unittest.TestCase):
    def test_required_conflict_reports_subset(self):
        sol = typeset(spec([
            item("A", 2, required=True, placements=((0, 4),)),
            item("B", 5, required=True, placements=((0, 4),)),
            item("C", 20, required=True, placements=((0, 1),)),
        ], width=30))
        self.assertEqual(sol["status"], "infeasible")
        self.assertEqual(sol["reason"]["cause"], "required_items_conflict")
        # A and B overlap; C is unrelated and must not be blamed.
        self.assertEqual(sol["reason"]["items"], ["A", "B"])

    def test_no_feasible_placement_reports_item(self):
        sol = typeset(spec(
            [item("A", 2, required=True, placements=((9, 10),))],
        ))
        self.assertEqual(sol["status"], "infeasible")
        self.assertEqual(sol["reason"]["cause"], "no_feasible_placement")
        self.assertEqual(sol["reason"]["item"], "A")

    def test_optional_unprintable_is_skipped_not_infeasible(self):
        sol = typeset(spec([
            item("A", 2, required=True),
            item("B", 8, value=4, placements=((5, 6),)),  # outside width
        ]))
        self.assertEqual(sol["status"], "ok")
        self.assertEqual(sol["value"], 0)
        self.assertEqual(set(boxes(sol)), {"A"})


class TestLimits(unittest.TestCase):
    def test_too_many_items(self):
        items = [item(f"N{i}", i) for i in range(19)]
        with self.assertRaises(LineSpecError):
            typeset(spec(items, width=100))

    def test_too_many_placements(self):
        with self.assertRaises(LineSpecError):
            typeset(spec([item("A", 2, placements=((0, 1), (1, 2), (2, 3), (3, 4)))]))

    def test_optional_needs_positive_int_value(self):
        with self.assertRaises(LineSpecError):
            typeset(spec([item("A", 2, value=0)]))
        with self.assertRaises(LineSpecError):
            typeset(spec([item("A", 2, value=-3)]))

    def test_required_carries_no_value(self):
        with self.assertRaises(LineSpecError):
            typeset(spec([item("A", 2, required=True, value=4)]))

    def test_non_positive_placement_rejected(self):
        with self.assertRaises(LineSpecError):
            typeset(spec([item("A", 2, required=True, placements=((2, 2),))]))

    def test_duplicate_note_rejected(self):
        with self.assertRaises(LineSpecError):
            typeset(spec([item("A", 2), item("A", 3)]))


if __name__ == "__main__":
    unittest.main()
