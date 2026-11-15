"""Per-session evaluation, report files, plots and aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .accuracy import (
    Q_COLS,
    TCP_COLS,
    ibvs_metrics,
    kinematic_accuracy,
    path_metrics,
    phase_rows,
    pose_errors,
    tracking_availability,
    vision_accuracy,
)
from .performance import motion_metrics, recovery_metrics, safety_metrics, timing_metrics

# A session counts as successful when IBVS converged, the final flange pose is
# within these limits and there was no safety stop.
THRESHOLDS = {"position_error_mm": 5.0, "orientation_error_deg": 2.0}

SECTIONS = ("accuracy", "ibvs", "path", "vision", "timing", "motion", "safety")
SECTION_TITLES = {
    "accuracy": "Pose accuracy (flange vs goal)",
    "ibvs": "IBVS (image feature error)",
    "path": "Approach motion (deviation from a straight line)",
    "vision": "Vision",
    "timing": "Timing and performance",
    "motion": "Motion",
    "safety": "Safety",
}

KEY_METRICS = [
    ("accuracy.final_position_error_mm", "Final position error (mm)"),
    ("accuracy.final_orientation_error_deg", "Final orientation error (deg)"),
    ("ibvs.final_error_px", "Final IBVS error (px)"),
    ("ibvs.settling_time_s", "IBVS settling time (s)"),
    ("path.path_deviation_max_mm", "Approach path deviation, max (mm)"),
    ("vision.detection_error_mean_px", "Detection error, mean (px)"),
    ("vision.detection_rate", "Detection rate"),
    ("timing.execution_time_sim_s", "Execution time, simulation (s)"),
    ("timing.avg_loop_rate_hz", "Average loop rate (Hz)"),
    ("safety.violation_events", "Safety violation events"),
]


def clean(obj):
    """Convert numpy values and NaN/inf into strict-JSON friendly Python types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    return str(obj)


def flatten(data: dict, prefix: str = "", sep: str = ".") -> dict:
    out = {}
    for key, value in data.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name, sep))
        elif isinstance(value, (list, tuple)):
            out[name] = ", ".join(str(v) for v in value)
        else:
            out[name] = value
    return out


# ----------------------------------------------------------------------
# One session
# ----------------------------------------------------------------------
def _verdict(report: dict) -> dict:
    reasons = []
    if not report["ibvs"].get("converged"):
        reasons.append("IBVS did not converge")

    accuracy = report["accuracy"]
    position = accuracy.get("final_position_error_mm")
    orientation = accuracy.get("final_orientation_error_deg")
    if position is None or orientation is None:
        reasons.append("no goal pose in the metadata")
    else:
        if position > THRESHOLDS["position_error_mm"]:
            reasons.append(f"position error {position:.1f} mm > {THRESHOLDS['position_error_mm']} mm")
        if orientation > THRESHOLDS["orientation_error_deg"]:
            reasons.append(f"orientation error {orientation:.2f} deg > {THRESHOLDS['orientation_error_deg']} deg")

    safety = report["safety"]
    if safety.get("blocked_samples", 0) or safety.get("estop_samples", 0):
        reasons.append("safety stop during the run")
    return {"success": not reasons, "reasons": reasons}


def evaluate_session(manager, name: str, robot, workcell=None, detector=None,
                    vision_frames: int = 60, tags: dict | None = None) -> dict:
    """Compute all metrics of one recorded session.

    Raises ValueError / FileNotFoundError if the session cannot be evaluated.
    workcell + detector (optional) enable the detection-accuracy check.
    """
    meta = manager.load_metadata(name)
    df = manager.load_robot_log(name)

    missing = [c for c in ["t_sim", *Q_COLS, *TCP_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"robot_log.csv has no column(s) {missing}")
    if len(df) < 5:
        raise ValueError("fewer than 5 samples")

    tol = float((meta.get("config") or {}).get("ibvs_tol_px", 1.5))

    report = {
        "session": name,
        "region": meta.get("region"),
        "tags": dict(tags or {}),
        "samples": int(len(df)),
        "accuracy": kinematic_accuracy(df, meta, robot),
        "ibvs": ibvs_metrics(df, tol, meta.get("procedure_status")),
        "path": path_metrics(df),
        "vision": {},
        "timing": timing_metrics(df, meta),
        "motion": motion_metrics(df),
        "safety": safety_metrics(df),
    }
    availability = tracking_availability(df, meta)
    if availability is not None:
        report["vision"]["tracking_availability"] = availability
    if workcell is not None and detector is not None:
        report["vision"].update(vision_accuracy(df, meta, workcell, detector, vision_frames))

    report["verdict"] = _verdict(report)
    return clean(report)


# ----------------------------------------------------------------------
# Report files
# ----------------------------------------------------------------------
def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items()) or "—"
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value) or "—"
    return str(value)


def render_markdown(report: dict) -> str:
    verdict = report["verdict"]
    lines = [
        f"# Evaluation: {report['session']}", "",
        f"- Region: {_fmt(report.get('region'))}",
        f"- Samples: {report['samples']}",
        f"- Result: **{'SUCCESS' if verdict['success'] else 'FAILED'}**",
    ]
    for reason in verdict["reasons"]:
        lines.append(f"  - {reason}")
    for section in SECTIONS:
        data = report.get(section)
        if not data:
            continue
        lines += ["", f"## {SECTION_TITLES[section]}", "", "| Metric | Value |", "|---|---|"]
        for key, value in data.items():
            lines.append(f"| {key.replace('_', ' ')} | {_fmt(value)} |")
    return "\n".join(lines) + "\n"


def plot_session(manager, name: str, robot, path) -> None:
    """Save a 2x2 summary figure: IBVS error, pose errors, TCP path, TCP speed."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    meta = manager.load_metadata(name)
    df = manager.load_robot_log(name)
    t0 = float(df["t_sim"].iloc[0])
    t = df["t_sim"].to_numpy(float) - t0

    servo = phase_rows(df, "servoing")
    t_servo = float(servo["t_sim"].iloc[0] - t0) if len(servo) else None

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    if "ibvs_error_px" in df.columns:
        ax.plot(t, df["ibvs_error_px"].to_numpy(float), label="IBVS error")
    tol = (meta.get("config") or {}).get("ibvs_tol_px")
    if tol:
        ax.axhline(tol, color="tab:red", ls="--", lw=1, label="tolerance")
    ax.set_title("IBVS feature error")
    ax.set_ylabel("RMS error (px)")
    ax.legend()

    ax = axes[0, 1]
    goal = meta.get("desired_flange_pose")
    if goal is not None:
        position, orientation = pose_errors(robot, df[Q_COLS].to_numpy(float), goal)
        ax.plot(t, position, label="position (mm)")
        ax.plot(t, orientation, label="orientation (deg)")
        ax.set_yscale("log")
        ax.legend()
    ax.set_title("Flange pose error")

    ax = axes[1, 0]
    P = df[TCP_COLS].to_numpy(float)
    ax.plot(P[:, 0], P[:, 1], lw=1.5)
    ax.plot(P[0, 0], P[0, 1], "o", mfc="none", label="start")
    ax.plot(P[-1, 0], P[-1, 1], "o", label="end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Flange path (top view)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()

    ax = axes[1, 1]
    dts = np.diff(t)
    valid = dts > 1e-9
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    ax.plot(t[1:][valid], steps[valid] / dts[valid])
    ax.set_title("Flange speed")
    ax.set_ylabel("m/s")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xlabel("time (s)")
        if t_servo is not None:
            ax.axvline(t_servo, color="gray", ls=":", lw=1)      # start of visual servoing

    fig.suptitle(f"{name}  (dotted line: start of visual servoing)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_session_report(manager, report: dict, robot, plots: bool = True) -> Path:
    """Write report.json, report.md and summary.png into <session>/evaluation/."""
    out = manager.path_of(report["session"]) / "evaluation"
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / "report.md").write_text(render_markdown(report))
    if plots:
        try:
            plot_session(manager, report["session"], robot, out / "summary.png")
        except Exception as exc:
            print(f"  (plot skipped: {exc})")
    return out


# ----------------------------------------------------------------------
# Several sessions
# ----------------------------------------------------------------------
def aggregate(reports: list, trials: list | None = None):
    """Statistics over many sessions. Sessions with an injected fault are not
    counted in the accuracy statistics (they are expected to stop early).

    Returns (summary dict, DataFrame with one flat row per session).
    """
    counted = [r for r in reports if r.get("tags", {}).get("kind") != "fault"]
    table = pd.DataFrame([flatten(r) for r in reports])
    summary = {
        "sessions": len(reports),
        "counted_sessions": len(counted),
        "fault_sessions": len(reports) - len(counted),
    }
    if counted:
        summary["success_rate"] = float(np.mean([bool(r["verdict"]["success"]) for r in counted]))
        frame = pd.DataFrame([flatten(r) for r in counted])
        stats = {}
        for column in frame.columns:
            if column.startswith("tags."):
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            stats[column] = {
                "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                "min": float(values.min()), "max": float(values.max()), "n": int(values.size),
            }
        summary["metrics"] = stats
    if trials is not None:
        summary["recovery"] = recovery_metrics(trials)
    return summary, table


def render_summary_markdown(summary: dict, trials: list | None = None) -> str:
    lines = [
        "# Evaluation summary", "",
        f"- Sessions: {summary['sessions']} "
        f"({summary['counted_sessions']} counted, {summary['fault_sessions']} with injected faults)",
    ]
    if "success_rate" in summary:
        lines.append(f"- Success rate: **{summary['success_rate'] * 100:.0f} %**")
    recovery = summary.get("recovery")
    if recovery and recovery.get("fault_runs"):
        lines.append(f"- Recovery success rate: **{recovery['recovery_success_rate'] * 100:.0f} %** "
                    f"({recovery['recovered_runs']}/{recovery['fault_runs']} runs)")

    metrics = summary.get("metrics", {})
    rows = [(label, metrics[key]) for key, label in KEY_METRICS if key in metrics]
    if rows:
        lines += ["", "| Metric | Mean | Std | Min | Max |", "|---|---|---|---|---|"]
        for label, s in rows:
            lines.append(f"| {label} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")

    if trials:
        lines += ["", "## Batch trials", "",
                "| Run | Region | Yaw (deg) | Noise | Fault | Status | Recovered |", "|---|---|---|---|---|---|---|"]
        for t in trials:
            fault = "yes" if t.get("fault_triggered") else ("planned, not reached" if t.get("fault_planned") else "no")
            lines.append(f"| {t['run']} | {t['region']} | {t['yaw_deg']} | {t['noise']} | {fault} | "
                        f"{t.get('status')} | {_fmt(t.get('recovered'))} |")
    return "\n".join(lines) + "\n"


def write_aggregate_report(summary: dict, table: pd.DataFrame, out_dir, trials: list | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(clean({"summary": summary, "trials": trials}), indent=2, allow_nan=False))
    table.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(render_summary_markdown(summary, trials))
    return out