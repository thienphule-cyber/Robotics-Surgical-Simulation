"""Transformations, forward/inverse kinematics, Jacobians and trajectories."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from control import (
    UR3E_HOME_Q,
    adjoint,
    cartesian_to_joint_trajectory,
    cartesian_trajectory,
    clip_to_joint_limits,
    forward_kinematics,
    inverse_kinematics,
    jacobian_base,
    jacobian_camera,
    jacobian_tool,
    joint_trajectory,
    make_pose,
    pose_error,
    project_points,
    scalar_profile,
    skew,
    within_joint_limits,
)

CONFIGS = [
    UR3E_HOME_Q.copy(),
    np.array([0.4, -1.2, -1.6, -1.2, 1.4, 0.3]),
    np.array([-0.5, -1.8, -1.0, -1.5, 1.57, -0.4]),
]
IDS = ["home", "config_a", "config_b"]


# ---------------------------------------------------------------- transformations
def test_skew_matches_cross_product():
    a = np.array([0.3, -1.2, 0.7])
    b = np.array([1.0, 0.5, -0.2])
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_adjoint_is_a_homomorphism():
    T1 = make_pose([0.1, 0.2, 0.3], [10, 20, 30])
    T2 = make_pose([-0.1, 0.05, 0.2], [-5, 15, 40])
    assert np.allclose(adjoint(T1 * T2), adjoint(T1) @ adjoint(T2), atol=1e-9)


def test_pose_error():
    T = make_pose([0.1, 0.2, 0.3], [0, 0, 0])
    assert np.allclose(pose_error(T, T), 0.0, atol=1e-12)

    shifted = SE3.Trans(0.01, -0.02, 0.03) * T
    err = pose_error(T, shifted)
    assert np.allclose(err[:3], [0.01, -0.02, 0.03], atol=1e-12)
    assert np.allclose(err[3:], 0.0, atol=1e-12)

    rotated = SE3.Rz(np.deg2rad(30)) * T
    assert np.allclose(pose_error(T, rotated)[3:], [0, 0, np.deg2rad(30)], atol=1e-9)


def test_project_points():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    pixels, depths = project_points(K, SE3(), np.array([[0, 0, 1.0], [0.1, 0, 1.0], [0, 0, -1.0]]))
    assert np.allclose(pixels[0], [320, 240])
    assert np.allclose(pixels[1], [370, 240])
    assert depths[0] == pytest.approx(1.0)
    assert np.all(np.isnan(pixels[2]))           # behind the camera


# ---------------------------------------------------------------- FK / IK
def test_home_pose_is_within_reach(robot):
    assert np.linalg.norm(forward_kinematics(robot, UR3E_HOME_Q).t) < 0.6


@pytest.mark.parametrize("q", CONFIGS, ids=IDS)
def test_ik_recovers_the_pose(robot, q):
    T = forward_kinematics(robot, q)
    q_sol, ok = inverse_kinematics(robot, T, q0=q + 0.05, n_random_seeds=3)
    assert ok
    T_sol = forward_kinematics(robot, q_sol)
    assert np.linalg.norm(T_sol.t - T.t) < 1e-4
    assert np.linalg.norm(pose_error(T_sol, T)[3:]) < 1e-3


def test_ik_reports_unreachable_targets(robot):
    q, ok = inverse_kinematics(robot, SE3.Trans(2.0, 0.0, 0.3), q0=UR3E_HOME_Q, n_random_seeds=0)
    assert not ok and q is None


def test_joint_limits(robot):
    assert within_joint_limits(robot, UR3E_HOME_Q)
    q = UR3E_HOME_Q.copy()
    q[0] = 7.0                                    # more than 2*pi
    assert not within_joint_limits(robot, q)
    assert within_joint_limits(robot, clip_to_joint_limits(robot, q))


# ---------------------------------------------------------------- Jacobians
def _numeric_jacobian_base(robot, q, eps=1e-6):
    T0 = robot.fkine(q)
    J = np.zeros((6, 6))
    for i in range(6):
        dq = q.copy()
        dq[i] += eps
        T1 = robot.fkine(dq)
        v = (T1.t - T0.t) / eps
        w = Rotation.from_matrix(T1.R @ T0.R.T).as_rotvec() / eps
        J[:, i] = np.r_[v, w]
    return J


def _numeric_body_twist(robot, q, T_tool_cam, eps=1e-6):
    T0 = robot.fkine(q) * T_tool_cam
    J = np.zeros((6, 6))
    for i in range(6):
        dq = q.copy()
        dq[i] += eps
        T1 = robot.fkine(dq) * T_tool_cam
        v = T0.R.T @ (T1.t - T0.t) / eps
        w = Rotation.from_matrix(T0.R.T @ T1.R).as_rotvec() / eps
        J[:, i] = np.r_[v, w]
    return J


@pytest.mark.parametrize("q", CONFIGS, ids=IDS)
def test_base_jacobian_matches_finite_differences(robot, q):
    assert np.allclose(jacobian_base(robot, q), _numeric_jacobian_base(robot, q), atol=1e-4)


@pytest.mark.parametrize("q", CONFIGS, ids=IDS)
def test_tool_jacobian_matches_finite_differences(robot, q):
    numeric = _numeric_body_twist(robot, q, SE3())
    assert np.allclose(jacobian_tool(robot, q), numeric, atol=1e-4)


@pytest.mark.parametrize("q", CONFIGS, ids=IDS)
def test_camera_jacobian_matches_finite_differences(robot, q):
    T_tool_cam = SE3.Trans(0.05, 0.0, 0.02) * SE3.Ry(np.deg2rad(-17))
    numeric = _numeric_body_twist(robot, q, T_tool_cam)
    assert np.allclose(jacobian_camera(robot, q, T_tool_cam), numeric, atol=1e-4)


# ---------------------------------------------------------------- trajectories
def test_quintic_profile_boundary_conditions():
    T = 2.0
    t = np.linspace(0, T, 201)
    s, sd, sdd = scalar_profile(t, T, "quintic")
    assert s[0] == pytest.approx(0.0) and s[-1] == pytest.approx(1.0)
    assert sd[0] == pytest.approx(0.0) and sd[-1] == pytest.approx(0.0)
    assert sdd[0] == pytest.approx(0.0) and sdd[-1] == pytest.approx(0.0)
    assert np.all(np.diff(s) >= -1e-12)
    assert s[100] == pytest.approx(0.5)
    assert sd.max() == pytest.approx(1.875 / T, rel=1e-3)


def test_trapezoidal_profile():
    T = 2.0
    t = np.linspace(0, T, 401)
    s, sd, _ = scalar_profile(t, T, "trapezoidal", blend=0.25)
    assert s[0] == pytest.approx(0.0) and s[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(s) >= -1e-12)
    assert sd.max() * T == pytest.approx(1.0 / 0.75, rel=1e-3)


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        scalar_profile(np.linspace(0, 1, 5), 1.0, "bogus")


def test_joint_trajectory():
    q0, q1 = UR3E_HOME_Q, UR3E_HOME_Q + 0.5
    traj = joint_trajectory(q0, q1, duration=1.0, dt=0.01)
    assert np.allclose(traj.q[0], q0) and np.allclose(traj.q[-1], q1)
    assert np.allclose(traj.qd[0], 0.0) and np.allclose(traj.qd[-1], 0.0)
    numeric_qd = np.gradient(traj.q, traj.t, axis=0)
    assert np.allclose(numeric_qd, traj.qd, atol=1e-2)


def test_cartesian_trajectory_is_a_straight_line():
    T0 = make_pose([0.3, 0.0, 0.3], [0, 0, 0])
    T1 = make_pose([0.4, 0.1, 0.2], [0, 0, 30])
    traj = cartesian_trajectory(T0, T1, duration=1.0, dt=0.01)
    assert np.allclose(traj.poses[0].A, T0.A) and np.allclose(traj.poses[-1].A, T1.A, atol=1e-9)
    assert np.allclose(traj.twists[0], 0.0) and np.allclose(traj.twists[-1], 0.0, atol=1e-9)
    mid = traj.poses[len(traj) // 2].t
    assert np.allclose(mid, (T0.t + T1.t) / 2, atol=1e-9)


def test_cartesian_to_joint_trajectory_follows_the_path(robot):
    T0 = forward_kinematics(robot, UR3E_HOME_Q)
    goal = SE3.Trans(0.0, 0.0, 0.05) * T0
    ctraj = cartesian_trajectory(T0, goal, duration=1.0, dt=0.05)
    traj = cartesian_to_joint_trajectory(robot, ctraj, q0=UR3E_HOME_Q)
    for q, pose in zip(traj.q, ctraj.poses):
        assert np.linalg.norm(forward_kinematics(robot, q).t - pose.t) < 1e-3
    assert np.max(np.abs(np.diff(traj.q, axis=0))) < 0.1       # continuous joint path


def test_cartesian_to_joint_trajectory_fails_when_unreachable(robot):
    T0 = forward_kinematics(robot, UR3E_HOME_Q)
    ctraj = cartesian_trajectory(T0, SE3.Trans(2.0, 0.0, 0.3), duration=0.2, dt=0.1)
    with pytest.raises(ValueError):
        cartesian_to_joint_trajectory(robot, ctraj, q0=UR3E_HOME_Q)