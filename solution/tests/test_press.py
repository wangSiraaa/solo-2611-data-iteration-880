"""Tests for the press room: sheets, proofs, lines, plates, durability.

Run:  python3 -m unittest discover -s solution/tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chainage_catalog import (  # noqa: E402
    CatalogRegistry,
    RouteCatalog,
    Section,
    load_catalog_registry,
    open_db,
    persist_catalog,
)
from inspection_library import (  # noqa: E402
    Asset,
    InspectionLibrary,
    ensure_inspection_schema,
    load_all_reports,
    load_assets,
    persist_assets,
    persist_report,
)
from panel_catalog import (  # noqa: E402
    PanelCatalog,
    ensure_panel_schema,
    load_panel_revisions,
    persist_panel_revision,
)
from press_room import (  # noqa: E402
    LineConflictError,
    PlateConflictError,
    PressRoom,
    SheetConflictError,
    SheetError,
    ensure_press_schema,
    load_line_issues,
    load_plates,
    load_sheets,
    persist_line_issue,
    persist_plate,
    persist_sheet,
)
from typeset_bench import LineSpecError  # noqa: E402


def make_stack(with_r2=False):
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
    if with_r2:
        reg.revise_section_label_start("r2", "L", "P2", 330)
    assets = [
        Asset("A", "L", Fraction(350)),
        Asset("B", "L", Fraction(470)),
        Asset("C", "L", Fraction(440)),
        Asset("P", "L", Fraction(2)),
        Asset("Q", "L", Fraction(5)),
    ]
    lib = InspectionLibrary(reg, assets)
    panels = PanelCatalog(reg)
    room = PressRoom(lib, panels)
    return reg, lib, panels, room


def make_reports(lib):
    lib.resolve_and_inspect(
        {"job_id": "J-PQ", "route_id": "L",
         "spans": [{"section_id": "P1", "label_start": 0, "label_end": 10}]},
        "R1",
    )
    lib.resolve_and_inspect(
        {"job_id": "J-B", "route_id": "L",
         "spans": [{"section_id": "P2", "label_start": 350, "label_end": 390}]},
        "R2",
    )


def define_g1(panels):
    panels.define_panel(
        "G1", "g1",
        [
            {"strip_id": "S-pq", "catalog_revision": "r1", "section_id": "P1",
             "label_start": 0, "label_end": 10, "paper_start": 0, "paper_end": 10},
            {"strip_id": "S-dup-p1", "catalog_revision": "r1",
             "section_id": "P1", "label_start": 340, "label_end": 360,
             "paper_start": 0, "paper_end": 10},
            {"strip_id": "S-dup-p2", "catalog_revision": "r1",
             "section_id": "P2", "label_start": 340, "label_end": 360,
             "paper_start": 0, "paper_end": 10},
        ],
    )


def sheet_t1():
    return {
        "sheet_id": "T1", "report_id": "R1", "panel_id": "G1",
        "panel_revision": "g1",
        "entries": [
            {"note_id": "N1", "strip_id": "S-pq", "asset_id": "P"},
            {"note_id": "N2", "strip_id": "S-pq", "asset_id": "Q"},
        ],
    }


def line_ln1():
    return {
        "line_id": "LN1", "sheet_id": "T1", "width": 10,
        "items": [
            {"note_id": "N1", "required": True, "placements": [[0, 3]]},
            {"note_id": "N2", "required": False, "value": 4,
             "placements": [[0, 3]]},
        ],
    }


class TestSheets(unittest.TestCase):
    def setUp(self):
        self.reg, self.lib, self.panels, self.room = make_stack()
        make_reports(self.lib)
        define_g1(self.panels)

    def test_accepted_anchors(self):
        sheet = self.room.issue_sheet(sheet_t1())
        self.assertEqual(
            {e.note_id: e.anchor for e in sheet.accepted},
            {"N1": Fraction(2), "N2": Fraction(5)},
        )
        self.assertEqual(sheet.rejected, ())

    def test_reject_reasons_are_concrete(self):
        sheet = self.room.issue_sheet({
            "sheet_id": "T9", "report_id": "R2", "panel_id": "G1",
            "panel_revision": "g1",
            "entries": [
                # B@470 is not in S-dup-p1's physical window [340,360).
                {"note_id": "E1", "strip_id": "S-dup-p1", "asset_id": "B"},
                # A is not covered by report R2 at all.
                {"note_id": "E2", "strip_id": "S-dup-p1", "asset_id": "A"},
                # No such strip in the panel.
                {"note_id": "E3", "strip_id": "S-nope", "asset_id": "B"},
            ],
        })
        reasons = {e.note_id: e.reason for e in sheet.rejected}
        self.assertEqual(
            reasons,
            {"E1": "outside_window", "E2": "facility_not_in_report",
             "E3": "unknown_strip"},
        )
        self.assertEqual(sheet.accepted, ())

    def test_revision_mismatch_reason(self):
        # A strip bound to r2 vs a report resolved under r1.
        self.reg.revise_section_label_start("r2", "L", "P2", 330)
        self.panels.define_panel(
            "G1", "g2",
            [{"strip_id": "S-b2", "catalog_revision": "r2", "section_id": "P2",
              "label_start": 350, "label_end": 390,
              "paper_start": 0, "paper_end": 10}],
        )
        sheet = self.room.issue_sheet({
            "sheet_id": "T8", "report_id": "R2", "panel_id": "G1",
            "panel_revision": "g2",
            "entries": [{"note_id": "M1", "strip_id": "S-b2", "asset_id": "B"}],
        })
        self.assertEqual(sheet.rejected[0].reason, "revision_mismatch")
        self.assertIn("r2", sheet.rejected[0].detail)
        self.assertIn("r1", sheet.rejected[0].detail)

    def test_same_label_two_sections_give_different_anchors(self):
        self.lib.resolve_and_inspect(
            {"job_id": "J-DUP", "route_id": "L",
             "spans": [
                 {"section_id": "P1", "label_start": 340, "label_end": 360},
                 {"section_id": "P2", "label_start": 340, "label_end": 360},
             ]},
            "R3",
        )
        sheet = self.room.issue_sheet({
            "sheet_id": "T3", "report_id": "R3", "panel_id": "G1",
            "panel_revision": "g1",
            "entries": [
                {"note_id": "N5", "strip_id": "S-dup-p1", "asset_id": "A"},
                {"note_id": "N6", "strip_id": "S-dup-p2", "asset_id": "C"},
            ],
        })
        anchors = {e.note_id: e.anchor for e in sheet.accepted}
        # Same label window [340,360), different sections -> different
        # physical windows -> different anchors.
        self.assertEqual(anchors, {"N5": Fraction(5), "N6": Fraction(0)})

    def test_unknown_report_rejects_whole_sheet(self):
        with self.assertRaises(SheetError):
            self.room.issue_sheet({
                "sheet_id": "TX", "report_id": "NOPE", "panel_id": "G1",
                "entries": [{"note_id": "a", "strip_id": "S-pq",
                             "asset_id": "P"}],
            })

    def test_sheet_id_content_addressed(self):
        s1 = self.room.issue_sheet(sheet_t1())
        s2 = self.room.issue_sheet(sheet_t1())  # idempotent
        self.assertEqual(s1.fingerprint(), s2.fingerprint())
        other = sheet_t1()
        other["entries"] = [
            {"note_id": "N1", "strip_id": "S-pq", "asset_id": "P"}
        ]
        with self.assertRaises(SheetConflictError):
            self.room.issue_sheet(other)

    def test_duplicate_note_id_rejected(self):
        bad = sheet_t1()
        bad["entries"].append(
            {"note_id": "N1", "strip_id": "S-pq", "asset_id": "Q"}
        )
        with self.assertRaises(SheetError):
            self.room.issue_sheet(bad)


class TestVerifyAndReprint(unittest.TestCase):
    def test_offline_recheck_survives_paper_and_catalog_revisions(self):
        reg, lib, panels, room = make_stack()
        make_reports(lib)
        define_g1(panels)
        sheet = room.issue_sheet(sheet_t1())
        fp = sheet.fingerprint()
        # Paper revision (panel g2) and catalog revision (r2) arrive later.
        panels.define_panel(
            "G1", "g2",
            [{"strip_id": "S-pq", "catalog_revision": "r1", "section_id": "P1",
              "label_start": 0, "label_end": 10, "paper_start": 0,
              "paper_end": 8}],
        )
        reg.revise_section_label_start("r2", "L", "P2", 330)
        result = room.verify_sheet("T1")
        self.assertTrue(result["verified"])
        self.assertEqual(result["fingerprint"], fp)
        self.assertEqual(result["anchors_checked"], 2)
        # The reprint view still carries the projection basis.
        view = room.sheet_view("T1")
        basis = view["accepted"][0]["basis"]
        self.assertEqual(basis["catalog_revision"], "r1")
        self.assertEqual(basis["scale"], "1/1")


class TestLinesAndPlates(unittest.TestCase):
    def setUp(self):
        self.reg, self.lib, self.panels, self.room = make_stack()
        make_reports(self.lib)
        define_g1(self.panels)
        self.room.issue_sheet(sheet_t1())

    def test_typeset_public_operation(self):
        issue = self.room.typeset_line(line_ln1())
        self.assertEqual(issue.solution["status"], "ok")
        self.assertEqual(issue.solution["value"], 4)
        self.assertEqual(issue.panel_revision, "g1")

    def test_line_id_spec_conflict(self):
        self.room.typeset_line(line_ln1())
        changed = line_ln1()
        changed["width"] = 12
        with self.assertRaises(LineConflictError):
            self.room.typeset_line(changed)

    def test_unknown_note_rejected(self):
        bad = line_ln1()
        bad["items"] = [
            {"note_id": "GHOST", "required": True, "placements": [[0, 3]]}
        ]
        with self.assertRaises(LineSpecError) as ctx:
            self.room.typeset_line(bad)
        self.assertIn("GHOST", str(ctx.exception))

    def test_freeze_and_reissue_only_affected_lines(self):
        room = self.room
        room.typeset_line(line_ln1())
        ln2 = line_ln1()
        ln2["line_id"] = "LN2"
        ln2["forbidden"] = [[7, 8]]
        room.typeset_line(ln2)
        # A line on a different strip that the paper change will not touch.
        self.panels.define_panel(
            "G1", "g1b",
            [
                {"strip_id": "S-pq", "catalog_revision": "r1",
                 "section_id": "P1", "label_start": 0, "label_end": 10,
                 "paper_start": 0, "paper_end": 8},  # changed
                {"strip_id": "S-dup-p1", "catalog_revision": "r1",
                 "section_id": "P1", "label_start": 340, "label_end": 360,
                 "paper_start": 0, "paper_end": 10},  # unchanged
            ],
        )
        plate = room.freeze_plate("PL1", "2026-11-20", ["LN1", "LN2"])
        result = room.reissue_plate("PL1", "g1b", "PL2", "2026-11-21")
        self.assertEqual(sorted(result["affected"]), ["LN1", "LN2"])
        self.assertEqual(result["unaffected"], [])
        # Re-issued LN1: anchors 8/5 and 4 -> boxes overlap -> only N1.
        new_issue = room.get_issue("LN1")
        self.assertEqual(new_issue.issue_seq, 2)
        self.assertEqual(new_issue.panel_revision, "g1b")
        self.assertEqual(new_issue.solution["value"], 0)
        # Frozen plate is immutable.
        with self.assertRaises(PlateConflictError):
            room.freeze_plate("PL1", "2026-11-21", ["LN1"])

    def test_unaffected_line_carries_frozen_issue(self):
        room = self.room
        room.typeset_line(line_ln1())
        self.panels.define_panel(
            "G1", "g1b",
            [{"strip_id": "S-pq", "catalog_revision": "r1",
              "section_id": "P1", "label_start": 0, "label_end": 10,
              "paper_start": 0, "paper_end": 10}],  # identical mapping
        )
        room.freeze_plate("PL1", "b1", ["LN1"])
        result = room.reissue_plate("PL1", "g1b", "PL2", "b2")
        self.assertEqual(result["affected"], [])
        self.assertEqual(result["unaffected"], ["LN1"])
        self.assertEqual(room.get_issue("LN1").issue_seq, 1)  # no re-typeset


class TestPressDurability(unittest.TestCase):
    def test_file_db_reopen_and_recheck(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "press.db")

            # --- first open: build everything, persist ---
            reg, lib, panels, room = make_stack()
            make_reports(lib)
            define_g1(panels)
            sheet = room.issue_sheet(sheet_t1())
            issue = room.typeset_line(line_ln1())
            plate = room.freeze_plate("PL1", "2026-11-20", ["LN1"])

            conn = open_db(path)
            ensure_inspection_schema(conn)
            ensure_panel_schema(conn)
            ensure_press_schema(conn)
            persist_catalog(conn, reg.get("r1"))
            persist_assets(conn, list(lib.assets))
            for rep in lib._reports.values():
                persist_report(conn, rep)
            for prev in panels.revisions("G1"):
                persist_panel_revision(conn, prev)
            persist_sheet(conn, sheet)
            persist_line_issue(conn, issue)
            persist_plate(conn, room, plate)
            conn.close()

            # --- second open: rebuild from storage only ---
            conn2 = open_db(path)
            ensure_inspection_schema(conn2)
            ensure_panel_schema(conn2)
            ensure_press_schema(conn2)
            reg2 = load_catalog_registry(conn2)
            lib2 = InspectionLibrary(reg2, load_assets(conn2))
            for rid, rep in load_all_reports(conn2).items():
                lib2._reports[rid] = rep
            panels2 = PanelCatalog(reg2)
            for prev in load_panel_revisions(conn2):
                panels2.define_panel(prev.panel_id, prev.panel_revision,
                                     prev.strips)
            room2 = PressRoom(lib2, panels2)
            room2._sheets.update(load_sheets(conn2))
            room2._lines.update(load_line_issues(conn2))
            room2._plates.update(load_plates(conn2))

            # The old sheet's proof re-checks offline after reopen.
            result = room2.verify_sheet("T1")
            self.assertTrue(result["verified"])
            self.assertEqual(result["fingerprint"], sheet.fingerprint())
            # The frozen line issue survived exactly.
            back = room2.get_issue("LN1", 1)
            self.assertEqual(back.fingerprint(), issue.fingerprint())
            self.assertEqual(back.solution["value"], 4)
            # And the plate still points at it.
            self.assertEqual(room2._plates["PL1"].lines, (("LN1", 1),))
            conn2.close()


if __name__ == "__main__":
    unittest.main()
