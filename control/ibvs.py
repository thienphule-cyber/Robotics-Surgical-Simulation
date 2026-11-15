"""Image-Based Visual Servoing (IBVS) with point features.

Control law (Chaumette & Hutchinson):
    e = s - s*                    feature error in normalised image coords
    v_c = -lambda * L^+ * e       camera twist [vx vy vz wx wy wz] in the
                                CAMERA frame
The twist is then sent to the robot with
    MotionController.command_twist(v_c, frame="camera", T_tool_cam=...)
so it always passes through the safety monitor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraIntrinsics, PinholeCamera
from .jacobian import damped_pinv
from .kinematics import forward_kinematics
from .safety import SafetyState


def interaction_matrix(xy, Z) -> np.ndarray:
    """Image Jacobian (2N x 6) of N point features.

    xy : (N, 2) normalised image coordinates (x, y).
    Z  : (N,) or scalar depth of each point in the camera frame.
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    n = len(xy)
    Z = np.broadcast_to(np.asarray(Z, dtype=float), (n,))
    L = np.zeros((2 * n, 6))
    for i in range(n):
        x, y = xy[i]
        z = Z[i]
        L[2 * i] = [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y]
        L[2 * i + 1] = [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x]
    return L


@dataclass
class IBVSResult:
    twist: np.ndarray        # (6,) camera twist in the camera frame
    error: np.ndarray        # (2n,) feature error in normalised coordinates
    error_px: float          # RMS feature distance error in pixels
    converged: bool
    n_features: int
    status: str              # "running", "converged" or "insufficient_features"
    L: np.ndarray | None = None


def desired_from_pose(camera: PinholeCamera, T_world_cam_desired, points_world):
    """Desired feature positions and depths seen from the desired camera pose.

    Raises ValueError if some point is not visible from that pose.
    """
    uv, depth, visible = camera.project(T_world_cam_desired, points_world)
    if not np.all(visible):
        raise ValueError("Not all target points are visible from the desired pose")
    return uv, depth


class IBVSController:
    def __init__(self, intrinsics: CameraIntrinsics, gain: float = 0.6,
                depth_mode: str = "mean", damping: float = 0.01,
                v_max: float = 0.08, w_max: float = 0.4,
                tol_px: float = 1.5, min_features: int = 3):
        """
        depth_mode : which interaction matrix to use
            "current": L at the current features (needs current depth)
            "desired": L at the desired features (only needs desired depth)
            "mean":    average of both (usually the best behaviour)
        v_max, w_max : limits on the camera linear (m/s) and angular (rad/s) speed.
        tol_px : convergence threshold on the RMS pixel error.
        """
        if depth_mode not in ("current", "desired", "mean"):
            raise ValueError("depth_mode must be 'current', 'desired' or 'mean'")
        self.intr = intrinsics
        self.gain = gain
        self.depth_mode = depth_mode
        self.damping = damping
        self.v_max = v_max
        self.w_max = w_max
        self.tol_px = tol_px
        self.min_features = min_features
        self.desired_uv = None
        self.desired_depth = None

    def set_desired(self, desired_uv, desired_depth):
        uv = np.asarray(desired_uv, dtype=float).reshape(-1, 2)
        self.desired_uv = uv
        self.desired_depth = np.broadcast_to(
            np.asarray(desired_depth, dtype=float), (len(uv),)).copy()

    def compute(self, uv, depth=None) -> IBVSResult:
        """Compute the camera twist from the current feature positions.

        uv    : (N, 2) current pixels; NaN rows are ignored (lost features).
        depth : (N,) current depth of each feature (simulation ground truth).
                If None, the desired depth is used instead.
        """
        if self.desired_uv is None:
            raise RuntimeError("Call set_desired() before compute()")

        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        valid = np.all(np.isfinite(uv), axis=1)
        n_valid = int(valid.sum())
        if n_valid < self.min_features:
            return IBVSResult(np.zeros(6), np.full(2 * len(uv), np.nan),
                            float("nan"), False, n_valid, "insufficient_features")

        s = self.intr.pixel_to_normalized(uv[valid])
        s_star = self.intr.pixel_to_normalized(self.desired_uv[valid])
        err = (s - s_star).ravel()

        px = uv[valid] - self.desired_uv[valid]
        rms_px = float(np.sqrt(np.mean(np.sum(px ** 2, axis=1))))

        Z_star = self.desired_depth[valid]
        Z_cur = Z_star if depth is None else np.asarray(depth, dtype=float)[valid]
        L_des = interaction_matrix(s_star, Z_star)
        L_cur = interaction_matrix(s, Z_cur)
        L = {"current": L_cur, "desired": L_des, "mean": 0.5 * (L_cur + L_des)}[self.depth_mode]

        if rms_px < self.tol_px:
            return IBVSResult(np.zeros(6), err, rms_px, True, n_valid, "converged", L)

        twist = -self.gain * (damped_pinv(L, self.damping) @ err)

        # Uniform scaling keeps the direction of motion.
        v_norm = np.linalg.norm(twist[:3])
        w_norm = np.linalg.norm(twist[3:])
        factor = min(1.0,
                    self.v_max / v_norm if v_norm > 1e-12 else 1.0,
                    self.w_max / w_norm if w_norm > 1e-12 else 1.0)
        twist = twist * factor

        return IBVSResult(twist, err, rms_px, False, n_valid, "running", L)


def run_ibvs(motion, camera: PinholeCamera, detector, tracker,
            ibvs: IBVSController, points_world, max_steps: int = 500,
            use_true_depth: bool = True, on_frame=None):
    """Closed visual servoing loop (eye-in-hand camera).

    Each step: render image -> detect -> track -> IBVS -> robot command
    (through the safety monitor).

    Parameters
    ----------
    motion : MotionController
    on_frame : optional callable(image, uv, desired_uv, result) called every
        step, for example to show the camera panel or write a video frame.

    Returns (status, log). status is one of "converged", "blocked",
    "lost_features" or "max_steps". log is a list of dicts for evaluation.
    """
    points_world = np.asarray(points_world, dtype=float)
    ids = range(len(points_world))
    log = []

    for step in range(max_steps):
        T_world_cam = camera.world_pose(forward_kinematics(motion.robot, motion.q))
        image = camera.render(T_world_cam, points_world)

        tracker.update(detector.detect(image))
        uv = tracker.positions(ids)
        depth = camera.depths(T_world_cam, points_world) if use_true_depth else None

        result = ibvs.compute(uv, depth)
        log.append({
            "t": motion.t,
            "q": motion.q,
            "uv": uv.copy(),
            "error_px": result.error_px,
            "twist": result.twist.copy(),
            "status": result.status,
        })
        if on_frame is not None:
            on_frame(image, uv, ibvs.desired_uv, result)

        if result.converged:
            return "converged", log
        if result.status == "insufficient_features":
            return "lost_features", log

        report = motion.command_twist(result.twist, frame="camera",
                                    T_tool_cam=camera.T_mount_cam)
        if report is not None and report.state in (SafetyState.BLOCKED,
                                                SafetyState.ESTOP):
            return "blocked", log

    return "max_steps", log