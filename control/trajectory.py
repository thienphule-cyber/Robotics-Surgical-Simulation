"""Trajectory generation: joint-space and Cartesian, with velocity profiling.

Profiles
--------
* "quintic":     smooth 5th-order polynomial (zero velocity and acceleration
                at both ends).
* "trapezoidal": linear segment with parabolic blends (LSPB), constant
                cruise velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from .kinematics import inverse_kinematics
from .transformations import to_se3


@dataclass
class JointTrajectory:
    """Sampled joint trajectory. All arrays share the first dimension N."""
    t: np.ndarray    # (N,)   time stamps [s]
    q: np.ndarray    # (N, n) joint positions [rad]
    qd: np.ndarray   # (N, n) joint velocities [rad/s]
    qdd: np.ndarray  # (N, n) joint accelerations [rad/s^2]

    def __len__(self) -> int:
        return len(self.t)


@dataclass
class CartesianTrajectory:
    """Sampled Cartesian trajectory (poses and twists in the base frame)."""
    t: np.ndarray        # (N,)
    poses: list          # list[SE3], length N
    twists: np.ndarray   # (N, 6) [vx, vy, vz, wx, wy, wz]

    def __len__(self) -> int:
        return len(self.t)


def _time_vector(duration: float, dt: float) -> np.ndarray:
    """Uniform time vector from 0 to duration (inclusive)."""
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    n = max(int(round(duration / dt)), 1)
    return np.linspace(0.0, duration, n + 1)


def scalar_profile(t: np.ndarray, duration: float, profile: str = "quintic",
                blend: float = 0.25):
    """Normalised scalar profile s(t) going from 0 to 1.

    Returns (s, sd, sdd): position, velocity and acceleration arrays.
    `blend` is the fraction of the duration used for each parabolic blend
    in the trapezoidal profile (0 < blend <= 0.5).
    """
    tau = np.clip(t / duration, 0.0, 1.0)

    if profile == "quintic":
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
        sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
        return s, sd, sdd

    if profile == "trapezoidal":
        if not 0.0 < blend <= 0.5:
            raise ValueError("blend must be in (0, 0.5]")
        vmax = 1.0 / (1.0 - blend)   # in normalised time units
        acc = vmax / blend
        s = np.zeros_like(tau)
        sd = np.zeros_like(tau)
        sdd = np.zeros_like(tau)

        up = tau < blend
        mid = (tau >= blend) & (tau <= 1.0 - blend)
        down = tau > 1.0 - blend

        s[up] = 0.5 * acc * tau[up] ** 2
        sd[up] = acc * tau[up]
        sdd[up] = acc

        s[mid] = 0.5 * acc * blend**2 + vmax * (tau[mid] - blend)
        sd[mid] = vmax

        s[down] = 1.0 - 0.5 * acc * (1.0 - tau[down]) ** 2
        sd[down] = acc * (1.0 - tau[down])
        sdd[down] = -acc

        return s, sd / duration, sdd / duration**2

    raise ValueError(f"Unknown profile '{profile}' (use 'quintic' or 'trapezoidal')")


def joint_trajectory(q_start, q_end, duration: float, dt: float,
                    profile: str = "quintic") -> JointTrajectory:
    """Point-to-point trajectory in joint space."""
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    t = _time_vector(duration, dt)
    s, sd, sdd = scalar_profile(t, duration, profile)

    dq = q_end - q_start
    q = q_start + s[:, None] * dq
    qd = sd[:, None] * dq
    qdd = sdd[:, None] * dq
    return JointTrajectory(t, q, qd, qdd)


def cartesian_trajectory(T_start, T_end, duration: float, dt: float,
                        profile: str = "quintic") -> CartesianTrajectory:
    """Straight-line trajectory in Cartesian space.

    Position is interpolated linearly; orientation is interpolated along the
    shortest rotation (constant rotation axis).
    """
    Ts = to_se3(T_start)
    Tg = to_se3(T_end)
    t = _time_vector(duration, dt)
    s, sd, _ = scalar_profile(t, duration, profile)

    dp = Tg.t - Ts.t
    rotvec = Rotation.from_matrix(Tg.R @ Ts.R.T).as_rotvec()

    poses = []
    for si in s:
        R = Rotation.from_rotvec(si * rotvec).as_matrix() @ Ts.R
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = Ts.t + si * dp
        poses.append(SE3(T, check=False))

    twists = np.zeros((len(t), 6))
    twists[:, :3] = sd[:, None] * dp
    twists[:, 3:] = sd[:, None] * rotvec
    return CartesianTrajectory(t, poses, twists)


def cartesian_to_joint_trajectory(robot, ctraj: CartesianTrajectory,
                                q0=None) -> JointTrajectory:
    """Convert a Cartesian trajectory to joint space using IK.

    Each waypoint is solved with the previous solution as the seed so the
    joint path stays continuous. Raises ValueError if IK fails somewhere
    (target out of reach or near a singularity).
    """
    q_prev = np.asarray(robot.q if q0 is None else q0, dtype=float)
    q_list = []
    for i, pose in enumerate(ctraj.poses):
        q, ok = inverse_kinematics(robot, pose, q0=q_prev, n_random_seeds=0)
        if not ok:
            raise ValueError(f"IK failed at waypoint {i}/{len(ctraj.poses) - 1}")
        q_list.append(q)
        q_prev = q

    q = np.vstack(q_list)
    qd = np.gradient(q, ctraj.t, axis=0)
    qdd = np.gradient(qd, ctraj.t, axis=0)
    return JointTrajectory(ctraj.t.copy(), q, qd, qdd)