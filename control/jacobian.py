"""Jacobian utilities for the UR3e, including the camera-frame Jacobian for IBVS.

Twist ordering: [vx, vy, vz, wx, wy, wz].
"""

from __future__ import annotations

import numpy as np
import roboticstoolbox as rtb

from .transformations import adjoint, to_se3


def jacobian_base(robot: rtb.DHRobot, q) -> np.ndarray:
    """6x6 Jacobian expressed in the base frame."""
    return robot.jacob0(np.asarray(q, dtype=float))


def jacobian_tool(robot: rtb.DHRobot, q) -> np.ndarray:
    """6x6 Jacobian expressed in the end-effector (tool) frame."""
    return robot.jacobe(np.asarray(q, dtype=float))


def jacobian_camera(robot: rtb.DHRobot, q, T_tool_cam) -> np.ndarray:
    """6x6 Jacobian giving the camera twist expressed in the camera frame.

    Parameters
    ----------
    T_tool_cam : pose of the eye-in-hand camera expressed in the tool frame
        (fixed hand-eye transform).

    Relation:  V_cam = J_cam @ qdot
    """
    T_tool_cam = to_se3(T_tool_cam)
    T_cam_tool = T_tool_cam.inv()
    Je = jacobian_tool(robot, q)
    return adjoint(T_cam_tool) @ Je


def manipulability(J: np.ndarray) -> float:
    """Yoshikawa manipulability measure sqrt(det(J J^T))."""
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def min_singular_value(J: np.ndarray) -> float:
    """Smallest singular value of J (close to 0 near a singularity)."""
    return float(np.linalg.svd(J, compute_uv=False).min())


def is_near_singularity(J: np.ndarray, threshold: float = 0.02) -> bool:
    """True if the smallest singular value is below the threshold."""
    return min_singular_value(J) < threshold


def damped_pinv(J: np.ndarray, damping: float = 0.05) -> np.ndarray:
    """Damped least-squares pseudo-inverse: J^T (J J^T + lambda^2 I)^-1.

    The damping makes the inverse stable near singularities at the cost of
    a small tracking error.
    """
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(m))


def cartesian_to_joint_velocity(
    J: np.ndarray,
    twist: np.ndarray,
    damping: float = 0.05,
    adaptive: bool = True,
    sv_threshold: float = 0.05,
) -> np.ndarray:
    """Convert a Cartesian twist into joint velocities.

    If `adaptive` is True, the damping is increased smoothly as the smallest
    singular value drops below `sv_threshold` (Nakamura-Hanafusa method).
    """
    twist = np.asarray(twist, dtype=float).ravel()
    lam = damping
    if adaptive:
        s_min = min_singular_value(J)
        if s_min < sv_threshold:
            lam = damping * np.sqrt(1.0 - (s_min / sv_threshold) ** 2)
            lam = max(lam, 1e-6)
        else:
            lam = 0.0
    return damped_pinv(J, lam) @ twist


def limit_joint_velocity(qdot: np.ndarray, qdot_max) -> np.ndarray:
    """Scale qdot uniformly so no joint exceeds qdot_max (rad/s).

    Uniform scaling preserves the direction of motion (unlike clipping each
    joint separately, which would distort the end-effector path).
    """
    qdot = np.asarray(qdot, dtype=float)
    qdot_max = np.broadcast_to(np.asarray(qdot_max, dtype=float), qdot.shape)
    ratio = np.max(np.abs(qdot) / qdot_max)
    if ratio > 1.0:
        qdot = qdot / ratio
    return qdot