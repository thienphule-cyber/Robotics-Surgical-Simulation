"""Accuracy metrics computed from a recorded session (no simulation needed).

Inputs: the robot_log.csv table (pandas DataFrame) and metadata.json (dict)
of one session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

Q_COLS = [f"q{i}" for i in range(1, 7)]
TCP_COLS = ["tcp_x", "tcp_y", "tcp_z"]

# The tracked feature error contains measurement noise, so a single sample can sit
# 10-15 % above the tolerance even when the run has converged. Convergence is
# therefore judged on a 10-sample average within this factor of the tolerance
# (when the session metadata has `procedure_status`, that value is used instead).
CONVERGENCE_MARGIN = 1.25


def _num(x):
    """float, or None for None / NaN / inf."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def phase_rows(df, phase: str):
    """Rows of the log recorded during one procedure phase."""
    if "phase" not in df.columns:
        return df.iloc[0:0]
    return df[df["phase"] == phase]


# ----------------------------------------------------------------------
# Pose accuracy
# ----------------------------------------------------------------------
def pose_errors(robot, q, T_goal):
    """Flange position error (mm) and orientation error (deg) for every row of q."""
    A = np.asarray(robot.fkine(np.asarray(q, dtype=float)).A).reshape(-1, 4, 4)
    goal = np.asarray(T_goal, dtype=float)
    position = np.linalg.norm(A[:, :3, 3] - goal[:3, 3], axis=1) * 1000.0
    R_rel = goal[:3, :3][None, :, :] @ A[:, :3, :3].transpose(0, 2, 1)
    orientation = np.degrees(Rotation.from_matrix(R_rel).magnitude())
    return position, orientation


def kinematic_accuracy(df, meta, robot) -> dict:
    """Flange pose error with respect to the goal pose stored in the metadata."""
    goal = meta.get("desired_flange_pose")
    if goal is None:
        return {}
    position, orientation = pose_errors(robot, df[Q_COLS].to_numpy(float), goal)
    return {
        "initial_position_error_mm": _num(position[0]),
        "initial_orientation_error_deg": _num(orientation[0]),
        "final_position_error_mm": _num(position[-1]),
        "final_orientation_error_deg": _num(orientation[-1]),
    }


# ----------------------------------------------------------------------
# IBVS image-feature error
# ----------------------------------------------------------------------
def ibvs_metrics(df, tol_px: float, procedure_status: str | None = None) -> dict:
    servo = phase_rows(df, "servoing")
    out = {"tolerance_px": float(tol_px), "servo_samples": int(len(servo)), "converged": False}
    if servo.empty or "ibvs_error_px" not in servo.columns:
        return out

    e = servo["ibvs_error_px"].to_numpy(float)
    t = servo["t_sim"].to_numpy(float)
    finite = np.isfinite(e)
    e, t = e[finite], t[finite]
    if e.size == 0:
        return out

    window = min(10, e.size)
    smooth = pd.Series(e).rolling(window, min_periods=window).mean().to_numpy()
    limit = tol_px * CONVERGENCE_MARGIN
    below = smooth <= limit                      # False while the average is not defined yet

    if procedure_status is not None:
        converged = procedure_status == "converged"
    else:
        converged = bool(below[-1])

    settling = None
    if converged:                                # time after which the average stays under the limit
        above = np.where(~below)[0]
        first = 0 if above.size == 0 else int(above[-1]) + 1
        if first < e.size:
            settling = float(t[first] - t[0])

    decay = None                                 # exponential decay rate while above the tolerance
    mask = e > limit
    if mask.sum() >= 10:
        slope = np.polyfit(t[mask], np.log(e[mask]), 1)[0]
        decay = float(-slope)

    out.update({
        "converged": converged,
        "initial_error_px": _num(e[0]),
        "final_error_px": _num(e[-1]),
        "mean_error_px": _num(e.mean()),
        "rms_error_px": _num(np.sqrt(np.mean(e ** 2))),
        "max_error_px": _num(e.max()),
        "settling_time_s": settling,
        "decay_rate_per_s": decay,
    })
    return out


# ----------------------------------------------------------------------
# Motion: how straight the approach motion is
# ----------------------------------------------------------------------
def path_metrics(df) -> dict:
    """Deviation of the approach (positioning phase) from the straight line
    between its first and last positions. The planned path is a straight
    Cartesian line, so this measures the tracking quality of the motion."""
    positioning = phase_rows(df, "positioning")
    if len(positioning) < 3 or not set(TCP_COLS) <= set(positioning.columns):
        return {}
    P = positioning[TCP_COLS].to_numpy(float)
    d = P[-1] - P[0]
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return {}
    u = d / length
    v = P - P[0]
    deviation = np.linalg.norm(v - np.outer(v @ u, u), axis=1) * 1000.0
    travelled = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return {
        "approach_distance_mm": length * 1000.0,
        "path_deviation_max_mm": _num(deviation.max()),
        "path_deviation_rms_mm": _num(np.sqrt(np.mean(deviation ** 2))),
        "path_length_ratio": travelled / length,
    }


# ----------------------------------------------------------------------
# Vision
# ----------------------------------------------------------------------
def tracking_availability(df, meta):
    """Average fraction of the markers that were tracked during servoing."""
    n_markers = len(meta.get("marker_points") or [])
    servo = phase_rows(df, "servoing")
    if n_markers == 0 or servo.empty or "ibvs_n_features" not in servo.columns:
        return None
    return _num(servo["ibvs_n_features"].astype(float).mean() / n_markers)


def vision_accuracy(df, meta, workcell, detector, max_frames: int = 60) -> dict:
    """Detection accuracy. Some servoing frames are re-rendered from the
    recorded joint angles and run through the detector; the detections are
    compared with the exact projection of the markers (ground truth).

    `workcell` is a (headless) Workcell and `detector` a ColorMarkerDetector.
    The result assumes the camera settings are the same as during the run.
    """
    servo = phase_rows(df, "servoing")
    if servo.empty:
        servo = df
    region = meta.get("region")
    if region in workcell.cfg.regions:
        workcell.select_region(region)
    points = workcell.marker_points
    cam = workcell.eye_camera
    robot = workcell.robot

    Q = servo[Q_COLS].to_numpy(float)
    picks = np.unique(np.linspace(0, len(Q) - 1, min(max_frames, len(Q))).astype(int))

    errors = []
    expected = 0
    detected = 0
    for i in picks:
        T_cam = cam.world_pose(robot.fkine(Q[i]))
        pixels, _, visible = cam.project(T_cam, points)
        found = {d.id: d.uv for d in detector.detect(cam.render(T_cam, points))}
        for m in range(len(points)):
            if not visible[m]:
                continue
            expected += 1
            if m in found:
                detected += 1
                errors.append(float(np.linalg.norm(found[m] - pixels[m])))

    return {
        "frames_checked": int(len(picks)),
        "detection_rate": detected / expected if expected else None,
        "detection_error_mean_px": _num(np.mean(errors)) if errors else None,
        "detection_error_max_px": _num(np.max(errors)) if errors else None,
    }