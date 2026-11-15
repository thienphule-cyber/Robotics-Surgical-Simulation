"""Timing, motion quality, safety and recovery metrics."""

from __future__ import annotations

import numpy as np

from .accuracy import TCP_COLS, _num

QDOT_COLS = [f"qdot{i}" for i in range(1, 7)]


def _dt(df, meta) -> float:
    if meta.get("dt"):
        return float(meta["dt"])
    steps = np.diff(df["t_sim"].to_numpy(float))
    steps = steps[steps > 0]
    return float(np.median(steps)) if steps.size else 0.02


def timing_metrics(df, meta) -> dict:
    """Execution time (simulation and wall clock), loop rate, phase durations."""
    dt = _dt(df, meta)
    t = df["t_sim"].to_numpy(float)
    samples = int(len(df))

    sim = meta.get("duration_sim_s")
    sim = float(t[-1] - t[0]) if sim is None else float(sim)
    wall = meta.get("duration_wall_s")

    durations = {}
    if "phase" in df.columns:
        for phase, group in df.groupby("phase", sort=False):
            tt = group["t_sim"].to_numpy(float)
            durations[str(phase)] = float(tt[-1] - tt[0] + dt)

    out = {
        "execution_time_sim_s": _num(sim),
        "execution_time_wall_s": _num(wall),
        "servoing_time_s": durations.get("servoing"),
        "phase_durations_s": durations,
        "control_samples": samples,
        "control_period_s": dt,
    }
    if wall and wall > 0:
        out["avg_loop_rate_hz"] = samples / wall          # includes rendering and recording
        out["realtime_factor"] = sim / wall               # > 1: faster than real time
    cameras = meta.get("cameras") or {}
    if cameras:
        out["video_frames"] = {name: c.get("frames") for name, c in cameras.items()}
        if wall and wall > 0:
            out["video_fps_wall"] = {name: (c.get("frames") or 0) / wall
                                     for name, c in cameras.items()}
    return out


def motion_metrics(df) -> dict:
    """Peak speeds and acceleration of the joints and the flange."""
    out = {}
    if set(QDOT_COLS) <= set(df.columns):
        qdot = df[QDOT_COLS].to_numpy(float)
        out["peak_joint_speed_deg_s"] = _num(np.degrees(np.nanmax(np.abs(qdot))))

    P = df[TCP_COLS].to_numpy(float)
    dts = np.diff(df["t_sim"].to_numpy(float))
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    valid = dts > 1e-9
    if valid.sum() >= 2:
        speed = steps[valid] / dts[valid]
        out["path_length_m"] = _num(steps.sum())
        out["peak_tcp_speed_m_s"] = _num(speed.max())
        out["mean_tcp_speed_m_s"] = _num(speed.mean())
        out["peak_tcp_accel_m_s2"] = _num(np.abs(np.diff(speed) / dts[valid][1:]).max())
    return out


def safety_metrics(df) -> dict:
    """Counts of the safety states recorded at every control step."""
    if "safety_state" not in df.columns:
        return {}
    state = df["safety_state"].astype(str)
    hazard = state.isin(["blocked", "estop"])
    events = int((hazard & ~hazard.shift(1, fill_value=False)).sum())   # entries into a stop
    return {
        "samples_by_state": {k: int(v) for k, v in state.value_counts().items()},
        "warning_samples": int((state == "warning").sum()),
        "blocked_samples": int((state == "blocked").sum()),
        "estop_samples": int((state == "estop").sum()),
        "violation_events": events,
        "safe_fraction": float(1.0 - hazard.mean()),
    }


def recovery_metrics(trials) -> dict:
    """Recovery success rate over the batch trials in which an e-stop was injected."""
    faults = [t for t in trials if t.get("fault_triggered")]
    if not faults:
        return {"fault_runs": 0, "recovered_runs": 0, "recovery_success_rate": None}
    recovered = sum(1 for t in faults if t.get("recovered"))
    return {"fault_runs": len(faults), "recovered_runs": recovered,
            "recovery_success_rate": recovered / len(faults)}