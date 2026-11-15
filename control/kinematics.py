"""UR3e kinematics: model, forward kinematics and numerical inverse kinematics.

The model uses standard DH parameters of the Universal Robots UR3e and is
built with roboticstoolbox so it works directly with Swift.
"""

from __future__ import annotations

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from .transformations import to_se3

# --- UR3e standard DH parameters (metres / radians) ---
_D = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]
_A = [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0]
_ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]

# Joint limits: +/- 2*pi for every joint (UR3e hardware limits).
UR3E_QLIM = np.array([[-2 * np.pi, 2 * np.pi]] * 6)

# A safe, non-singular home configuration (elbow up, tool pointing down).
UR3E_HOME_Q = np.deg2rad([0.0, -90.0, -90.0, -90.0, 90.0, 0.0])


def build_ur3e(name: str = "UR3e") -> rtb.DHRobot:
    """Create the UR3e as an rtb.DHRobot (compatible with swift.Swift().add)."""
    links = [
        rtb.RevoluteDH(d=_D[i], a=_A[i], alpha=_ALPHA[i], qlim=UR3E_QLIM[i])
        for i in range(6)
    ]
    robot = rtb.DHRobot(links, name=name)
    robot.q = UR3E_HOME_Q.copy()
    return robot


def forward_kinematics(robot: rtb.DHRobot, q) -> SE3:
    """Pose of the end-effector (flange) in the base frame for joints q."""
    return robot.fkine(np.asarray(q, dtype=float))


def within_joint_limits(robot: rtb.DHRobot, q, margin: float = 0.0) -> bool:
    """True if every joint is inside its limits (optionally shrunk by margin rad)."""
    q = np.asarray(q, dtype=float)
    lo = robot.qlim[0] + margin
    hi = robot.qlim[1] - margin
    return bool(np.all(q >= lo) and np.all(q <= hi))


def clip_to_joint_limits(robot: rtb.DHRobot, q) -> np.ndarray:
    """Clip a joint vector to the robot joint limits."""
    return np.clip(np.asarray(q, dtype=float), robot.qlim[0], robot.qlim[1])


def inverse_kinematics(
    robot: rtb.DHRobot,
    T_target,
    q0=None,
    n_random_seeds: int = 10,
    pos_tol: float = 1e-4,
    rng_seed: int | None = 0,
):
    """Numerical inverse kinematics (Levenberg-Marquardt) with multiple seeds.

    The solver first tries q0 (or the robot's current q), then random seeds
    inside the joint limits. Among the successful solutions, the one closest
    to q0 is returned so the motion stays smooth.

    Returns
    -------
    (q, success) : q is a (6,) array (or None if all attempts failed).
    """
    T_target = to_se3(T_target)
    q_ref = np.asarray(robot.q if q0 is None else q0, dtype=float)

    rng = np.random.default_rng(rng_seed)
    seeds = [q_ref]
    for _ in range(n_random_seeds):
        seeds.append(rng.uniform(robot.qlim[0], robot.qlim[1]))

    best_q = None
    best_dist = np.inf
    for seed in seeds:
        sol = robot.ikine_LM(T_target, q0=seed)
        if not sol.success:
            continue
        q = np.asarray(sol.q, dtype=float)
        if not within_joint_limits(robot, q):
            continue
        # Verify the solution with forward kinematics.
        err = np.linalg.norm(robot.fkine(q).t - T_target.t)
        if err > pos_tol:
            continue
        dist = np.linalg.norm(q - q_ref)
        if dist < best_dist:
            best_dist = dist
            best_q = q

    return best_q, best_q is not None