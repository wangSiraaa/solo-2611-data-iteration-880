"""Tests for the guide panel catalog (strips, projection, revisions).

Run:  python3 -m unittest discover -s solution/tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chainage_catalog import (  # noqa: E402
    CatalogRegistry,
    RouteCatalog,
    Section,
    open_db,
    persist_catalog,
)
from panel_catalog import (  # noqa: E402
    OutsideWindowError,
    PanelCatalog,
    PanelConflictError,
    PanelError,
    Strip,
    ensure_panel_schema,
    load_panel_revisions,
    persist_panel_revision,
)


def registry_r1() -> CatalogRegistry:
    reg = CatalogRegistry()
    reg.add(
        RouteCatalog(
            "r1", "L", 1000,
            [
                Section("P1", Fraction(0), Fraction(400), Fraction(0)),
                Section("P2", Fraction(400), Fraction(700), Fraction(300)),
                Section("P3", Fraction(700), Fraction(1000), Fraction(650)),
            ],
        )
    )
    return reg


def strip(**kw) -> Strip:
    base = dict(
        strip_id="S",
        catalog_revision="r1",
        section_id="P1",
        label_start=Fraction(0),
        label_end=Fraction(10),
        paper_start=Fraction(0),
        paper_end=Fraction(10),
    )
    base.update(kw)
    return Strip(**base)


class TestStripValidation(unittest.TestCase):
    def setUp(self):
        self.panels = PanelCatalog(registry_r1())

    def test_unknown_catalog_revision_rejected(self):
        with self.assertRaises(PanelError):
            self.panels.define_panel(
                "G", "g1", [strip(catalog_revision="rX")]
            )

    def test_unknown_section_rejected(self):
        with self.assertRaises(PanelError):
            self.panels.define_panel("G", "g1", [strip(section_id="PX")])

    def test_window_outside_section_rejected(self):
        with self.assertRaises(PanelError):
            self.panels.define_panel(
                "G", "g1", [strip(label_start=Fraction(395), label_end=Fraction(401))]
            )

    def test_window_end_may_equal_section_end_label(self):
        # P1 labels [0,400): upper bound 400 is a legal boundary.
        prev = self.panels.define_panel(
            "G", "g1",
            [strip(label_start=Fraction(390), label_end=Fraction(400))],
        )
        self.assertEqual(prev.strips[0].label_end, Fraction(400))

    def test_non_positive_window_rejected(self):
        with self.assertRaises(PanelError):
            self.panels.define_panel(
                "G", "g1", [strip(label_end=Fraction(0))]
            )

    def test_identical_paper_points_rejected(self):
        with self.assertRaises(PanelError):
            self.panels.define_panel(
                "G", "g1", [strip(paper_end=Fraction(0))]
            )

    def test_duplicate_strip_id_rejected(self):
        with self.assertRaises(PanelError):
            self.panels.define_panel("G", "g1", [strip(), strip()])

    def test_panel_revision_content_addressed(self):
        self.panels.define_panel("G", "g1", [strip()])
        again = self.panels.define_panel("G", "g1", [strip()])  # idempotent
        self.assertEqual(again.fingerprint(), self.panels.get("G", "g1").fingerprint())
        with self.assertRaises(PanelConflictError):
            self.panels.define_panel(
                "G", "g1", [strip(paper_end=Fraction(8))]
            )

    def test_panel_revisions_are_independent_of_catalog_revisions(self):
        reg = registry_r1()
        panels = PanelCatalog(reg)
        panels.define_panel("G", "g1", [strip()])
        reg.revise_section_label_start("r2", "L", "P2", 330)
        panels.define_panel("G", "g2", [strip(paper_end=Fraction(8))])
        # Both panel revisions still resolve against r1 strips.
        self.assertEqual(panels.get("G", "g1").strips[0].paper_end, Fraction(10))
        self.assertEqual(panels.get("G", "g2").strips[0].paper_end, Fraction(8))
        self.assertEqual(panels.get("G").panel_revision, "g2")  # current


class TestProjection(unittest.TestCase):
    def setUp(self):
        self.panels = PanelCatalog(registry_r1())

    def test_forward_scale_one(self):
        self.panels.define_panel("G", "g1", [strip()])
        a = self.panels.anchor("G", "S", 2)
        self.assertEqual(a["anchor"], Fraction(2))
        self.assertEqual(a["basis"]["scale"], Fraction(1))
        self.assertFalse(a["basis"]["reversed"])

    def test_reverse_strip(self):
        self.panels.define_panel(
            "G", "g1", [strip(paper_start=Fraction(10), paper_end=Fraction(0))]
        )
        a = self.panels.anchor("G", "S", 2)
        self.assertEqual(a["anchor"], Fraction(8))
        self.assertEqual(a["basis"]["scale"], Fraction(-1))
        self.assertTrue(a["basis"]["reversed"])

    def test_proportional_exact_rationals(self):
        # The worked example: window [0,10) onto paper 0..8 -> scale 4/5.
        self.panels.define_panel("G", "g1", [strip(paper_end=Fraction(8))])
        self.assertEqual(self.panels.anchor("G", "S", 2)["anchor"], Fraction(8, 5))
        self.assertEqual(self.panels.anchor("G", "S", 5)["anchor"], Fraction(4))

    def test_physical_not_label_is_projected(self):
        # A strip on P2: label window [350,390) is physical [450,490) in r1.
        self.panels.define_panel(
            "G", "g1",
            [strip(section_id="P2", label_start=Fraction(350),
                   label_end=Fraction(390))],
        )
        a = self.panels.anchor("G", "S", 470)
        self.assertEqual(a["anchor"], Fraction(5))
        self.assertEqual(a["basis"]["physical_window"], (Fraction(450), Fraction(490)))
        self.assertEqual(a["basis"]["label_at_physical"], Fraction(370))

    def test_window_is_half_open(self):
        self.panels.define_panel("G", "g1", [strip()])
        with self.assertRaises(OutsideWindowError):
            self.panels.anchor("G", "S", 10)  # right end excluded
        self.assertEqual(self.panels.anchor("G", "S", 0)["anchor"], Fraction(0))

    def test_outside_window_reports_position_and_window(self):
        self.panels.define_panel("G", "g1", [strip()])
        with self.assertRaises(OutsideWindowError) as ctx:
            self.panels.anchor("G", "S", 12)
        msg = str(ctx.exception)
        self.assertIn("12/1", msg)
        self.assertIn("[0/1, 10/1)", msg)

    def test_anchor_uses_bound_revision_not_current(self):
        # The designer must not re-locate an old draft with today's labels:
        # a strip bound to r1 keeps mapping through r1 even after r2 exists.
        reg = registry_r1()
        panels = PanelCatalog(reg)
        panels.define_panel(
            "G", "g1",
            [strip(section_id="P2", label_start=Fraction(350),
                   label_end=Fraction(390))],
        )
        reg.revise_section_label_start("r2", "L", "P2", 330)
        a = panels.anchor("G", "S", 470)  # still r1: physical [450,490)
        self.assertEqual(a["anchor"], Fraction(5))
        self.assertEqual(a["basis"]["catalog_revision"], "r1")


class TestPanelPersistence(unittest.TestCase):
    def test_roundtrip(self):
        conn = open_db(":memory:")
        ensure_panel_schema(conn)
        panels = PanelCatalog(registry_r1())
        panels.define_panel(
            "G", "g1",
            [strip(), strip(strip_id="T", paper_start=Fraction(10),
                            paper_end=Fraction(0))],
        )
        persist_panel_revision(conn, panels.get("G", "g1"))
        persist_panel_revision(conn, panels.get("G", "g1"))  # idempotent
        back = load_panel_revisions(conn)
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0].fingerprint(), panels.get("G", "g1").fingerprint())
        self.assertTrue(back[0].strips[1].reversed)
        conn.close()


if __name__ == "__main__":
    unittest.main()
