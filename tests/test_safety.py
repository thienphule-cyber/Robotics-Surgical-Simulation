"""SafetyMonitor: limits, zones, e-stop, recovery."""

import numpy as np
import pytest

from control import (
    UR3E_HOME_Q,
    Box,
    SafetyConfig,
    SafetyMonitor,
    SafetyState,
    Sphere,
    forward_kinematics,
    jacobian_base,
)

DT = 0.02
HOME = UR3E_HOME_Q.copy()


def make_monitor(robot, **overrides):
    params = dict(floor_z=None, link_radius=0.0, qdot_max=1.0, tcp_speed_max=1.0)
    params.update(overrides)
    return SafetyMonitor(robot, SafetyConfig(**params))


def tcp_of(robot, q):
    return np.asarray(forward_kinematics(robot, q).t)


# ---------------------------------------------------------------- zones
def test_box_signed_distance():
    box = Box("b", [0, 0, 0], [1, 1, 1])
    d = box.signed_distance(np.array([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [1.0, 0.5, 0.5]]))
    assert d[0] == pytest.approx(-0.5)
    assert d[1] == pytest.approx(1.0)
    assert d[2] == pytest.approx(0.0)


def test_sphere_signed_distance():
    sphere = Sphere("s", [0, 0, 0], 1.0)
    d = sphere.signed_distance(np.array([[3.0, 0, 0], [0, 0, 0]]))
    assert d[0] == pytest.approx(2.0) and d[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------- static checks
def test_a_safe_configuration_has_no_violations(robot):
    report = make_monitor(robot).check_configuration(HOME)
    assert report.violations == [] and report.state is SafetyState.OK


def test_joint_limit_violation(robot):
    q = HOME.copy()
    q[0] = 2 * np.pi - 0.01                      # inside the 2 degree margin
    assert "joint 1 limit" in make_monitor(robot).check_configuration(q).violations


def test_workspace_boundary(robot):
    tcp = tcp_of(robot, HOME)
    inside = make_monitor(robot, workspace=Box("ws", tcp - 0.1, tcp + 0.1))
    assert inside.check_configuration(HOME).violations == []
    outside = make_monitor(robot, workspace=Box("ws", tcp + 0.1, tcp + 0.2))
    assert any("workspace" in v for v in outside.check_configuration(HOME).violations)


def test_restricted_zone(robot):
    tcp = tcp_of(robot, HOME)
    monitor = make_monitor(robot, restricted_zones=[Sphere("head", tcp, 0.02)])
    assert "restricted zone 'head'" in monitor.check_configuration(HOME).violations


def test_floor_contact(robot):
    monitor = make_monitor(robot, floor_z=10.0)
    assert "floor contact" in monitor.check_configuration(HOME).violations


# ---------------------------------------------------------------- velocity filter
def test_joint_speed_limit_scales_uniformly(robot):
    monitor = make_monitor(robot)
    qdot, report = monitor.filter_velocity(HOME, np.array([3.0, 0, 0, 0, 0, 0]), DT)
    assert qdot[0] == pytest.approx(1.0)
    assert np.allclose(qdot[1:], 0.0)
    assert report.velocity_scale == pytest.approx(1.0 / 3.0)


def test_tcp_speed_limit(robot):
    monitor = make_monitor(robot, tcp_speed_max=0.05)
    request = np.array([0.8, 0.5, 0.3, 0.0, 0.0, 0.0])
    qdot, _ = monitor.filter_velocity(HOME, request, DT)
    speed = np.linalg.norm((jacobian_base(robot, HOME) @ qdot)[:3])
    assert speed <= 0.05 + 1e-6
    assert np.dot(qdot, request) > 0                     # same direction


def test_motion_into_a_zone_is_blocked(robot):
    qdot_req = np.array([0.5, 0, 0, 0, 0, 0])
    tcp_now = tcp_of(robot, HOME)
    tcp_next = tcp_of(robot, HOME + qdot_req * DT)
    assert np.linalg.norm(tcp_next - tcp_now) > 0.002     # sanity: the TCP really moves

    zone = Sphere("ahead", tcp_next, 0.001)
    monitor = make_monitor(robot, restricted_zones=[zone])
    assert monitor.check_configuration(HOME).violations == []

    qdot, report = monitor.filter_velocity(HOME, qdot_req, DT)
    assert np.allclose(qdot, 0.0)
    assert report.state is SafetyState.BLOCKED
    assert monitor.violation_count >= 1


# ---------------------------------------------------------------- e-stop and recovery
def test_unsafe_current_state_latches_the_estop(robot):
    tcp = tcp_of(robot, HOME)
    monitor = make_monitor(robot, restricted_zones=[Sphere("z", tcp, 0.05)])
    qdot, report = monitor.filter_velocity(HOME, np.zeros(6), DT)
    assert report.state is SafetyState.ESTOP and monitor.estopped
    assert np.allclose(qdot, 0.0)

    qdot, report = monitor.filter_velocity(HOME, np.ones(6) * 0.1, DT)    # stays stopped
    assert np.allclose(qdot, 0.0) and report.state is SafetyState.ESTOP


def test_estop_can_only_be_released_in_a_safe_pose(robot):
    tcp = tcp_of(robot, HOME)
    monitor = make_monitor(robot, restricted_zones=[Sphere("z", tcp, 0.05)])
    monitor.trigger_estop("test")
    assert not monitor.reset_estop(HOME) and monitor.estopped

    monitor.cfg.restricted_zones.clear()
    assert monitor.reset_estop(HOME) and not monitor.estopped


def test_manual_estop_blocks_motion(robot):
    monitor = make_monitor(robot)
    monitor.trigger_estop("manual")
    qdot, report = monitor.filter_velocity(HOME, np.ones(6) * 0.2, DT)
    assert np.allclose(qdot, 0.0) and report.state is SafetyState.ESTOP


def test_recovery_returns_to_the_last_safe_pose(robot):
    monitor = make_monitor(robot)
    assert monitor.recovery_step(HOME, DT) == (None, False)      # nothing recorded yet

    monitor.filter_velocity(HOME, np.zeros(6), DT)               # records the safe pose
    q = HOME + 0.2
    for _ in range(200):
        qdot, done = monitor.recovery_step(q, DT, speed_scale=0.3)
        if done:
            break
        assert np.max(np.abs(qdot)) <= 0.3 * monitor.cfg.qdot_max + 1e-9
        q = q + qdot * DT
    assert done and np.linalg.norm(q - HOME) < 1e-3