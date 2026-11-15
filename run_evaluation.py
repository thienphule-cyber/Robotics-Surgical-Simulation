#!/usr/bin/env python3
"""run_evaluation.py - evaluate recorded sessions, or run headless trials first.

Examples
--------
python run_evaluation.py --latest                    newest session in recordings/
python run_evaluation.py --session NAME              one or more named sessions
python run_evaluation.py --all                       every session + a summary
python run_evaluation.py --batch 10                  10 headless trials, then evaluate
python run_evaluation.py --batch 10 --faults 0.3     ... with injected e-stops (recovery test)

Evaluation only reads robot_log.csv and metadata.json, so it is fast. Reports go to
<session>/evaluation/ and, for several sessions, to a summary folder.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from control import ColorMarkerDetector, build_ur3e
from evaluation import (
    aggregate,
    evaluate_session,
    write_aggregate_report,
    write_session_report,
)
from recording import SessionManager


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate recorded surgical-simulation sessions")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true", help="evaluate the newest session")
    group.add_argument("--session", action="append", metavar="NAME", help="evaluate this session (repeatable)")
    group.add_argument("--all", action="store_true", help="evaluate every session")
    group.add_argument("--batch", type=int, metavar="N", help="run N headless trials, then evaluate them")

    p.add_argument("--root", default="recordings", help="folder with the sessions")
    p.add_argument("--no-plots", action="store_true", help="skip the summary figures (faster)")
    p.add_argument("--no-vision", action="store_true", help="skip the detection-accuracy check (faster)")
    p.add_argument("--vision-frames", type=int, default=60, help="frames re-rendered per session")

    p.add_argument("--faults", type=float, default=0.0, help="batch: fraction of trials with an injected e-stop")
    p.add_argument("--seed", type=int, default=0, help="batch: random seed")
    p.add_argument("--yaw-max", type=float, default=10.0, help="batch: tool yaw is random in +/- this many degrees")
    p.add_argument("--noise-max", type=float, default=2.0, help="batch: camera noise is random in [0, this]")
    p.add_argument("--max-steps", type=int, default=1500, help="batch: maximum IBVS steps per trial")
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# Batch of headless trials
# ----------------------------------------------------------------------
def _inject_estop(app, rng) -> dict:
    """Trigger an emergency stop at a random frame of the visual servoing."""
    trigger_at = int(rng.integers(20, 120))
    original = app.on_frame
    count = {"frames": 0}

    def on_frame(image, uv, desired_uv, result):
        original(image, uv, desired_uv, result)
        count["frames"] += 1
        if count["frames"] == trigger_at and not app.monitor.estopped:
            app.monitor.trigger_estop("injected fault", app.ctrl.t)

    app.on_frame = on_frame
    return count


def _recover_and_rerun(app):
    """Release the e-stop (back to the last safe pose) and run the procedure again."""
    app._reset_estop()
    if app.monitor.estopped:
        return False, None
    result = app.run_procedure()
    session = app.rec.session.name if app.rec.session else None
    return result == "converged", session


def run_batch(args, batch_dir: Path) -> list:
    import main as sim                                   # the simulation entry point
    from workcell import WorkcellConfig

    rng = np.random.default_rng(args.seed)
    regions = list(WorkcellConfig().regions)
    trials = []

    for i in range(args.batch):
        trial = {
            "run": i,
            "region": regions[i % len(regions)],
            "yaw_deg": round(float(rng.uniform(-args.yaw_max, args.yaw_max)), 2),
            "noise": round(float(rng.uniform(0.0, args.noise_max)), 2),
            "fault_planned": bool(rng.random() < args.faults),
            "fault_triggered": False,
            "recovered": None,
            "status": None,
            "session": None,
            "recovery_session": None,
        }
        argv = ["--headless", "--no-dashboard", "--record", "--record-dir", str(batch_dir),
                "--region", trial["region"], "--tool-yaw", str(trial["yaw_deg"]),
                "--noise", str(trial["noise"]), "--max-steps", str(args.max_steps)]

        app = sim.Application(sim.parse_args(argv))
        t0 = time.perf_counter()
        try:
            app.setup()
            app.rec.capture = None                       # log only: no video, much faster
            if trial["fault_planned"]:
                _inject_estop(app, rng)
            trial["status"] = app.run_procedure()
            trial["session"] = app.rec.session.name if app.rec.session else None
            if trial["fault_planned"] and app.monitor.estopped:
                trial["fault_triggered"] = True
                trial["recovered"], trial["recovery_session"] = _recover_and_rerun(app)
        finally:
            app.shutdown()

        extra = ""
        if trial["fault_triggered"]:
            extra = f", recovered: {trial['recovered']}"
        print(f"  trial {i + 1}/{args.batch}: {trial['region']:<8} yaw {trial['yaw_deg']:>6} "
            f"noise {trial['noise']:<4} -> {trial['status']}{extra} ({time.perf_counter() - t0:.1f} s)")
        trials.append(trial)
    return trials


def jobs_from_trials(trials: list) -> list:
    jobs = []
    for t in trials:
        if t["session"]:
            kind = "fault" if t["fault_triggered"] else "nominal"
            jobs.append((t["session"], {"kind": kind, "trial": t["run"]}))
        if t["recovery_session"]:
            jobs.append((t["recovery_session"], {"kind": "recovery", "trial": t["run"]}))
    return jobs


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def select_sessions(manager, args) -> list:
    if args.all:
        return [s["name"] for s in manager.list_sessions()]
    if args.latest:
        latest = manager.latest()
        return [latest] if latest else []
    return list(args.session)


def make_vision_tools(args):
    """A headless Workcell + detector used to re-render frames (or (None, None))."""
    if args.no_vision:
        return None, None
    from workcell import Workcell
    workcell = Workcell(headless=True)
    return workcell, ColorMarkerDetector(n_markers=workcell.cfg.n_markers)


def _cell(report: dict, section: str, key: str, digits: int = 2) -> str:
    value = report.get(section, {}).get(key)
    return "-" if value is None else f"{value:.{digits}f}"


def print_table(reports: list):
    print()
    print(f"{'session':<34} {'kind':<9} {'result':<8} {'pos mm':>7} {'ori deg':>8} "
        f"{'ibvs px':>8} {'settle s':>9} {'sim s':>7} {'stops':>6}")
    for r in reports:
        kind = r.get("tags", {}).get("kind", "recorded")
        result = "OK" if r["verdict"]["success"] else "FAILED"
        stops = r["safety"].get("violation_events", 0)
        print(f"{r['session']:<34} {kind:<9} {result:<8} "
            f"{_cell(r, 'accuracy', 'final_position_error_mm'):>7} "
            f"{_cell(r, 'accuracy', 'final_orientation_error_deg'):>8} "
            f"{_cell(r, 'ibvs', 'final_error_px'):>8} "
            f"{_cell(r, 'ibvs', 'settling_time_s'):>9} "
            f"{_cell(r, 'timing', 'execution_time_sim_s'):>7} {stops:>6}")
    for r in reports:
        if not r["verdict"]["success"] and r.get("tags", {}).get("kind") != "fault":
            print(f"  {r['session']}: " + "; ".join(r["verdict"]["reasons"]))


def print_summary(summary: dict):
    print()
    if "success_rate" in summary:
        print(f"Success rate: {summary['success_rate'] * 100:.0f} % "
            f"({summary['counted_sessions']} counted session(s))")
    recovery = summary.get("recovery")
    if recovery and recovery.get("fault_runs"):
        print(f"Recovery success rate: {recovery['recovery_success_rate'] * 100:.0f} % "
            f"({recovery['recovered_runs']}/{recovery['fault_runs']})")
    metrics = summary.get("metrics", {})
    for key in ("accuracy.final_position_error_mm", "accuracy.final_orientation_error_deg",
                "ibvs.final_error_px", "ibvs.settling_time_s"):
        if key in metrics:
            s = metrics[key]
            print(f"  {key:<40} mean {s['mean']:.3f}  std {s['std']:.3f}  max {s['max']:.3f}")


# ----------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    trials = None
    if args.batch:
        batch_dir = Path(args.root) / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
        print(f"Running {args.batch} headless trial(s) into {batch_dir} ...")
        trials = run_batch(args, batch_dir)
        manager = SessionManager(batch_dir)
        jobs = jobs_from_trials(trials)
        out_dir = batch_dir
    else:
        manager = SessionManager(args.root)
        jobs = [(name, {"kind": "recorded"}) for name in select_sessions(manager, args)]
        out_dir = Path(args.root) / "evaluation_summary"

    if not jobs:
        print(f"No sessions found in '{manager.root}'. Record one first, for example:\n"
            "  python main.py --headless --once --no-dashboard --record")
        return 1

    robot = build_ur3e()
    workcell, detector = make_vision_tools(args)

    reports = []
    print(f"Evaluating {len(jobs)} session(s) ...")
    for name, tags in jobs:
        try:
            report = evaluate_session(manager, name, robot, workcell, detector,
                                    args.vision_frames, tags)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"  skipped {name}: {exc}")
            continue
        write_session_report(manager, report, robot, plots=not args.no_plots)
        reports.append(report)

    if workcell is not None:
        workcell.close()
    if not reports:
        print("Nothing could be evaluated.")
        return 1

    print_table(reports)
    if len(reports) > 1 or trials is not None:
        summary, table = aggregate(reports, trials)
        folder = write_aggregate_report(summary, table, out_dir, trials)
        print_summary(summary)
        print(f"\nSummary written to {folder}")
    else:
        print(f"\nReport written to {manager.path_of(reports[0]['session']) / 'evaluation'}")

    ok = all(r["verdict"]["success"] for r in reports if r.get("tags", {}).get("kind") != "fault")
    print(f"Done in {time.perf_counter() - started:.1f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())