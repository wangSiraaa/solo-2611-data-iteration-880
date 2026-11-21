"""CLI / orchestrator for guide panels, proof sheets and the typesetting bench.

Reads the materials in ``materials/731`` (route, assets, inspections,
panels, sheets, lines, operations), applies operations in order, persists
everything in SQLite (four separate schemas: catalogue, inspection facts,
panels, press room), and demonstrates:

* the worked example (strip [0,10) -> paper 0..10, P and Q touching boxes,
  the [7,8) forbidden zone, and the paper-end-8改版);
* a reverse (逆排) strip;
* two sections carrying the same label window;
* rejection reasons for bad annotation entries;
* daily plate freeze and re-issue of only the affected lines;
* offline re-check and reprint of an old sheet after a paper revision and
  a catalog revision.

Usage:
    python3 solution/run_press.py [--materials DIR] [--db PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from fractions import Fraction

from chainage_catalog import (
    CatalogRegistry,
    RouteCatalog,
    load_catalog_registry,
    open_db,
    persist_catalog,
)
from inspection_library import (
    Asset,
    InspectionLibrary,
    ensure_inspection_schema,
    load_all_reports,
    load_assets,
    persist_assets,
    persist_report,
)
from panel_catalog import (
    PanelCatalog,
    ensure_panel_schema,
    load_panel_revisions,
    persist_panel_revision,
)
from press_room import (
    PressRoom,
    ensure_press_schema,
    load_line_issues,
    load_plates,
    load_sheets,
    persist_line_issue,
    persist_plate,
    persist_sheet,
)
from run_demo import _read_json, _read_jsonl, _frac


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _x(value) -> str:
    """Render a Fraction or a stored 'n/d' string compactly."""
    if isinstance(value, Fraction):
        return _frac(value)
    return _frac(Fraction(str(value)))


def _box(box) -> str:
    return f"[{_x(box[0])}, {_x(box[1])})"


def print_panel(prev) -> None:
    print(f"  panel {prev.panel_id} revision {prev.panel_revision} "
          f"(fingerprint {prev.fingerprint()[:12]}…)")
    for s in prev.strips:
        direction = "逆排" if s.reversed else "顺排"
        print(
            f"    strip {s.strip_id}: catalog {s.catalog_revision} "
            f"{s.section_id} labels [{_x(s.label_start)}, {_x(s.label_end)}) "
            f"-> paper {_x(s.paper_start)}..{_x(s.paper_end)} ({direction})"
        )


def print_sheet(room: PressRoom, sheet_id: str) -> None:
    view = room.sheet_view(sheet_id)
    print(f"  sheet {sheet_id}  (report {view['report_id']}, panel "
          f"{view['panel_id']}/{view['panel_revision']}, "
          f"fingerprint {view['fingerprint'][:12]}…)")
    for e in view["accepted"]:
        b = e["basis"]
        print(
            f"    {e['note_id']}: {e['asset_id']}@physical {_x(e['physical'])} "
            f"-> anchor x={_x(e['anchor'])}  "
            f"[strip {e['strip_id']}, {b['catalog_revision']}/{b['section_id']} "
            f"physical {_box(b['physical_window'])} -> paper "
            f"{_x(b['paper_start'])}..{_x(b['paper_end'])}, "
            f"scale {_x(b['scale'])}, {'逆排' if b['reversed'] else '顺排'}]"
        )
    for e in view["rejected"]:
        print(f"    {e['note_id']}: REJECTED ({e['reason']}) {e['detail']}")


def print_solution(issue) -> None:
    sol = issue.solution
    print(f"  line {issue.line_id} issue {issue.issue_seq} "
          f"(sheet {issue.sheet_id}, panel revision {issue.panel_revision})")
    if sol["status"] == "infeasible":
        reason = sol["reason"]
        print(f"    INFEASIBLE ({reason['cause']}): {reason['detail']}")
        return
    print(f"    value {sol['value']}")
    for a in sol["assignment"]:
        print(f"    {a['note_id']} placement {a['placement']} "
              f"box {_box(a['box'])}")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class Application:
    """Holds in-memory state mirrored to SQLite."""

    MUTATING = {
        "resolve_and_inspect", "revise_chainage", "define_panel",
        "revise_panel", "issue_sheet", "typeset_line", "freeze_plate",
        "reissue_plate",
    }

    def __init__(self, registry, lib, panels, room, conn, materials_dir):
        self.registry = registry
        self.lib = lib
        self.panels = panels
        self.room = room
        self.conn = conn
        self.materials_dir = materials_dir
        self._persisted_panels = {
            (r["panel_id"], r["panel_revision"])
            for r in conn.execute("SELECT panel_id, panel_revision FROM panel_revision")
        }
        self._persisted_sheets = {
            r["sheet_id"] for r in conn.execute("SELECT sheet_id FROM sheet")
        }
        self._persisted_issues = {
            (r["line_id"], r["issue_seq"])
            for r in conn.execute("SELECT line_id, issue_seq FROM line_issue")
        }
        self._persisted_plates = {
            r["plate_id"] for r in conn.execute("SELECT plate_id FROM plate")
        }
        self._persisted_reports = {
            r["report_id"] for r in conn.execute("SELECT report_id FROM report")
        }
        self._persisted_revisions = {
            r["revision_id"] for r in conn.execute("SELECT revision_id FROM route_revision")
        }

    def _path(self, name: str) -> str:
        return os.path.join(self.materials_dir, name)

    # -- durable op-log (shared table from the inspection schema) -------
    def _op_done(self, op: dict) -> bool:
        h = hashlib.sha256(
            json.dumps(op, sort_keys=True).encode("utf-8")
        ).hexdigest()
        row = self.conn.execute(
            "SELECT 1 FROM applied_op WHERE op_hash = ?", (h,)
        ).fetchone()
        return row is not None

    def _mark_op_done(self, op: dict) -> None:
        h = hashlib.sha256(
            json.dumps(op, sort_keys=True).encode("utf-8")
        ).hexdigest()
        with self.conn:
            seq = self.conn.execute(
                "SELECT COALESCE(MAX(op_seq), 0) + 1 FROM applied_op"
            ).fetchone()[0]
            self.conn.execute(
                "INSERT INTO applied_op (op_seq, op_hash) VALUES (?, ?)",
                (seq, h),
            )

    # -- persistence of freshly created objects -------------------------
    def _persist_report(self, report) -> None:
        if report.report_id not in self._persisted_reports:
            persist_report(self.conn, report)
            self._persisted_reports.add(report.report_id)

    def _persist_panels(self) -> None:
        for key in self.panels._order:
            if key not in self._persisted_panels:
                persist_panel_revision(self.conn, self.panels._revisions[key])
                self._persisted_panels.add(key)

    def _persist_sheet(self, sheet) -> None:
        if sheet.sheet_id not in self._persisted_sheets:
            persist_sheet(self.conn, sheet)
            self._persisted_sheets.add(sheet.sheet_id)

    def _persist_issue(self, issue) -> None:
        key = (issue.line_id, issue.issue_seq)
        if key not in self._persisted_issues:
            persist_line_issue(self.conn, issue)
            self._persisted_issues.add(key)

    def _persist_plate(self, plate) -> None:
        if plate.plate_id not in self._persisted_plates:
            persist_plate(self.conn, self.room, plate)
            self._persisted_plates.add(plate.plate_id)

    # -- dispatch ---------------------------------------------------------
    def apply(self, op: dict) -> None:
        kind = op.get("op")
        if kind in self.MUTATING and self._op_done(op):
            print(f"\n(skipping already-applied op: {kind})")
            return
        handler = getattr(self, f"_op_{kind}", None)
        if handler is None:
            raise ValueError(f"unknown operation {kind!r}")
        handler(op)
        if kind in self.MUTATING:
            self._mark_op_done(op)

    # -- operations -------------------------------------------------------
    def _op_resolve_and_inspect(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        report_id = str(op["report_id"])
        current = self.registry.current()
        print(f"\n[resolve_and_inspect] report_id={report_id} "
              f"(revision {current.revision_id})")
        for label in payload.get("point_queries", []):
            result = current.resolve_label_point(label)
            if result["status"] == "ambiguous":
                parts = ", ".join(
                    f"{c['section_id']}@{_x(c['physical'])}"
                    for c in result["candidates"]
                )
                print(f"  point label {_x(result['label'])}: AMBIGUOUS: {parts}")
            else:
                print(f"  point label {_x(result['label'])}: {result['status']}")
        for job in payload.get("jobs", []):
            try:
                report = self.lib.resolve_and_inspect(job, report_id)
            except Exception as exc:
                print(f"  job {job.get('job_id')} NOT PUBLISHED: {exc}")
                continue
            self._persist_report(report)
            facs = ", ".join(
                f"{f.asset_id}@{_x(f.physical_position)}"
                for f in report.facilities
            ) or "(none)"
            print(f"  job {job['job_id']}: facilities covered: {facs}")

    def _op_revise_chainage(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        print(f"\n[revise_chainage] {payload['route_id']} "
              f"{payload['section_id']} label_start -> {payload['label_start']} "
              f"(revision {payload['revision']})")
        cat = self.registry.revise_section_label_start(
            revision_id=str(payload["revision"]),
            route_id=str(payload["route_id"]),
            section_id=str(payload["section_id"]),
            label_start=payload["label_start"],
        )
        if cat.revision_id not in self._persisted_revisions:
            persist_catalog(self.conn, cat)
            self._persisted_revisions.add(cat.revision_id)
        affected = self.lib.reports_affected_by(cat.revision_id)
        if affected:
            print(f"  reports now stale (kept as-is): {', '.join(affected)}")

    def _op_define_panel(self, op: dict) -> None:
        self._define_panel(op, "define_panel")

    def _op_revise_panel(self, op: dict) -> None:
        self._define_panel(op, "revise_panel")

    def _define_panel(self, op: dict, label: str) -> None:
        payload = _read_json(self._path(op["file"]))
        print(f"\n[{label}] {payload['panel_id']} "
              f"revision {payload['panel_revision']}")
        prev = self.panels.define_panel(
            payload["panel_id"], payload["panel_revision"], payload["strips"]
        )
        self._persist_panels()
        print_panel(prev)

    def _op_issue_sheet(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        print(f"\n[issue_sheet] {payload['sheet_id']} "
              f"(report {payload['report_id']})")
        sheet = self.room.issue_sheet(payload)
        self._persist_sheet(sheet)
        print_sheet(self.room, sheet.sheet_id)

    def _op_typeset_line(self, op: dict) -> None:
        payload = _read_json(self._path(op["file"]))
        print(f"\n[typeset_line] {payload['line_id']} "
              f"(sheet {payload['sheet_id']})")
        issue = self.room.typeset_line(payload)
        self._persist_issue(issue)
        print_solution(issue)

    def _op_freeze_plate(self, op: dict) -> None:
        print(f"\n[freeze_plate] {op['plate_id']} batch {op['batch']}")
        plate = self.room.freeze_plate(op["plate_id"], op["batch"], op["lines"])
        self._persist_plate(plate)
        for line_id, seq in plate.lines:
            issue = self.room.get_issue(line_id, seq)
            sol = issue.solution
            status = (f"value {sol['value']}" if sol["status"] == "ok"
                      else sol["status"])
            print(f"  frozen {line_id} issue {seq} ({status})")
        print(f"  plate fingerprint {self.room.plate_fingerprint(plate.plate_id)[:12]}…")

    def _op_reissue_plate(self, op: dict) -> None:
        print(f"\n[reissue_plate] {op['plate_id']} -> {op['new_plate_id']} "
              f"under panel revision {op['panel_revision']}")
        result = self.room.reissue_plate(
            op["plate_id"], op["panel_revision"], op["new_plate_id"], op["batch"]
        )
        for issue in result["new_issues"]:
            self._persist_issue(issue)
        plate = self.room._plates[result["plate_id"]]
        self._persist_plate(plate)
        for line_id in result["unaffected"]:
            print(f"  {line_id}: unchanged, frozen issue carried over")
        for issue in result["new_issues"]:
            print(f"  {issue.line_id}: anchors moved, re-typeset:")
            print_solution(issue)
        print(f"  new plate {result['plate_id']} batch {result['batch']} "
              f"(affected: {', '.join(result['affected']) or 'none'})")

    def _op_recheck_sheet(self, op: dict) -> None:
        sheet_id = str(op["sheet_id"])
        print(f"\n[recheck_sheet] {sheet_id} (offline re-check)")
        result = self.room.verify_sheet(sheet_id)
        verdict = "VERIFIED" if result["verified"] else "MISMATCH"
        print(f"  {verdict}: {result['anchors_checked']} anchors rebuilt from "
              f"panel {result['panel_revision']} + catalog "
              f"{result['catalog_revision']}; fingerprint "
              f"{result['fingerprint'][:12]}…")

    def _op_reprint_sheet(self, op: dict) -> None:
        sheet_id = str(op["sheet_id"])
        print(f"\n[reprint_sheet] {sheet_id} (复印旧单)")
        result = self.room.verify_sheet(sheet_id)
        if not result["verified"]:
            print("  PROOF MISMATCH -- refusing to reprint")
            return
        print_sheet(self.room, sheet_id)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap(materials_dir: str, db_path: str):
    conn = open_db(db_path)
    ensure_inspection_schema(conn)
    ensure_panel_schema(conn)
    ensure_press_schema(conn)

    route = _read_json(os.path.join(materials_dir, "route.json"))
    persist_catalog(conn, RouteCatalog.from_dict(route))
    registry = load_catalog_registry(conn)

    assets_data = _read_json(os.path.join(materials_dir, "assets.json"))
    persist_assets(conn, [Asset.from_dict(a) for a in assets_data["assets"]])
    lib = InspectionLibrary(registry, load_assets(conn))
    for rid, rep in load_all_reports(conn).items():
        lib._reports[rid] = rep

    panels = PanelCatalog(registry)
    for prev in load_panel_revisions(conn):
        panels.define_panel(prev.panel_id, prev.panel_revision, prev.strips)

    room = PressRoom(lib, panels)
    room._sheets.update(load_sheets(conn))
    room._lines.update(load_line_issues(conn))
    room._plates.update(load_plates(conn))
    return conn, registry, lib, panels, room


def run(materials_dir: str, db_path: str) -> None:
    conn, registry, lib, panels, room = bootstrap(materials_dir, db_path)
    app = Application(registry, lib, panels, room, conn, materials_dir)
    ops = _read_jsonl(os.path.join(materials_dir, "operations.jsonl"))

    print("=" * 78)
    print("Guide panels + proof sheets + typesetting bench")
    print("=" * 78)
    cat = registry.current()
    print(f"\nInitial catalog revision {cat.revision_id} of route "
          f"{cat.route_id} (length {_x(cat.length)})")

    for op in ops:
        app.apply(op)

    print("\n" + "=" * 78)
    print("Summary: the worked example, end to end")
    print("=" * 78)
    _summarize(room)


def _summarize(room: PressRoom) -> None:
    for line_id in ("LN1", "LN2"):
        try:
            issues = room._lines[line_id]
        except KeyError:
            continue
        for issue in issues:
            sol = issue.solution
            if sol["status"] != "ok":
                continue
            boxes = ", ".join(
                f"{a['note_id']} {_box(a['box'])}" for a in sol["assignment"]
            )
            print(f"  {issue.line_id} issue {issue.issue_seq} "
                  f"(panel {issue.panel_revision}): value {sol['value']} -- {boxes}")


def main(argv=None) -> int:
    default_materials = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "materials", "731",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", default=default_materials)
    parser.add_argument("--db", default=":memory:")
    args = parser.parse_args(argv)
    run(args.materials, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
