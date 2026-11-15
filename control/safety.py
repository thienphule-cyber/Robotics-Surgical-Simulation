"""Safety supervision: limits, restricted zones, collision checks, e-stop.

The SafetyMonitor sits between ANY velocity command (trajectory follower,
IBVS, manual jog...) and the robot. It always has the final word: it can
scale, block or zero a command, and it can latch an emergency stop.

Simplifications (documented on purpose)
---------------------------------------
* The robot is approximated by its link segments with a fixed radius.
* Workspace limits are checked on the tool centre point (TCP) only.
* Self-collision is NOT checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .jacobian import jacobian_base


class SafetyState(Enum):
    OK = "ok"
    WARNING = "warning"   # close to a zone/limit, motion slowed down
    BLOCKED = "blocked"   # command would cause a violation, motion refused
    ESTOP = "estop"       # emergency stop latched


@dataclass
class Box:
    """Axis-aligned box (workspace or restricted zone)."""
    name: str
    min_corner: np.ndarray
    max_corner: np.ndarray

    def __post_init__(self):
        self.min_corner = np.asarray(self.min_corner, dtype=float)
        self.max_corner = np.asarray(self.max_corner, dtype=float)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance: >0 outside the box, <0 inside."""
        p = np.atleast_2d(points)
        d = np.maximum(self.min_corner - p, p - self.max_corner)
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        return np.where(outside > 0.0, outside, d.max(axis=1))


@dataclass
class Sphere:
    """Spherical restricted zone."""
    name: str
    center: np.ndarray
    radius: float

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(points)
        return np.linalg.norm(p - self.center, axis=1) - self.radius


@dataclass
class SafetyConfig:
    qlim_margin: float = np.deg2rad(2.0)   # rad, shrinks the joint limits
    qdot_max: float = 1.0                  # rad/s per joint
    tcp_speed_max: float = 0.25            # m/s, linear tool speed limit
    workspace: Box | None = None           # TCP must stay inside
    restricted_zones: list = field(default_factory=list)  # keep-out zones
    floor_z: float | None = 0.0            # links must stay above this z (base frame)
    link_radius: float = 0.05              # m, robot thickness for collision checks
    warn_distance: float = 0.05            # m, slow down inside this clearance
    tool_length: float = 0.0               # m, instrument length along tool z
    samples_per_link: int = 5


@dataclass
class SafetyReport:
    state: SafetyState
    violations: list = field(default_factory=list)
    min_clearance: float = np.inf          # m, smallest clearance found
    velocity_scale: float = 1.0            # factor applied to the command


class SafetyMonitor:
    def __init__(self, robot, config: SafetyConfig | None = None):
        self.robot = robot
        self.cfg = config or SafetyConfig()
        self.estopped = False
        self.estop_reason = ""
        self.last_safe_q = None
        self.violations = []               # log of (time, message)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _robot_points(self, q):
        """Return (joint_points, tcp, sampled_points) in the base frame."""
        pts = [np.asarray(self.robot.base.t, dtype=float)]
        for T in self.robot.fkine_all(q):
            p = np.asarray(T.t, dtype=float)
            if np.linalg.norm(p - pts[-1]) > 1e-9:
                pts.append(p)

        T_tool = self.robot.fkine(q)
        tcp = np.asarray(T_tool.t, dtype=float) + T_tool.R[:, 2] * self.cfg.tool_length
        if np.linalg.norm(tcp - pts[-1]) > 1e-9:
            pts.append(tcp)

        joint_pts = np.array(pts)
        n = self.cfg.samples_per_link
        samples = []
        for a, b in zip(joint_pts[:-1], joint_pts[1:]):
            for w in np.linspace(0.0, 1.0, n):
                samples.append((1 - w) * a + w * b)
        return joint_pts, tcp, np.array(samples)

    # ------------------------------------------------------------------
    # State checking
    # ------------------------------------------------------------------
    def check_configuration(self, q) -> SafetyReport:
        """Check a joint configuration against all static safety rules."""
        q = np.asarray(q, dtype=float)
        cfg = self.cfg
        violations = []
        clearances = []

        # Joint limits (with margin)
        lo = self.robot.qlim[0] + cfg.qlim_margin
        hi = self.robot.qlim[1] - cfg.qlim_margin
        for i in np.where((q < lo) | (q > hi))[0]:
            violations.append(f"joint {i + 1} limit")

        joint_pts, tcp, samples = self._robot_points(q)

        # Workspace boundary (TCP only)
        if cfg.workspace is not None:
            sd = float(cfg.workspace.signed_distance(tcp)[0])
            clearances.append(-sd)
            if sd > 0.0:
                violations.append(f"workspace boundary ({cfg.workspace.name})")

        # Restricted zones (all link samples, inflated by link radius)
        for zone in cfg.restricted_zones:
            c = float(zone.signed_distance(samples).min()) - cfg.link_radius
            clearances.append(c)
            if c < 0.0:
                violations.append(f"restricted zone '{zone.name}'")

        # Floor / table contact (joint origins and TCP, base point excluded)
        if cfg.floor_z is not None:
            check_pts = np.vstack([joint_pts[1:], tcp])
            c = float(check_pts[:, 2].min()) - cfg.floor_z - cfg.link_radius
            clearances.append(c)
            if c < 0.0:
                violations.append("floor contact")

        min_c = min(clearances) if clearances else np.inf
        if violations:
            state = SafetyState.BLOCKED
        elif min_c < cfg.warn_distance:
            state = SafetyState.WARNING
        else:
            state = SafetyState.OK
        return SafetyReport(state, violations, min_c)

    # ------------------------------------------------------------------
    # Velocity filtering (the safety override)
    # ------------------------------------------------------------------
    def filter_velocity(self, q, qdot, dt: float, t: float = 0.0):
        """Return (safe_qdot, report) for a requested joint velocity."""
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float).copy()
        zero = np.zeros_like(qdot)

        if self.estopped:
            return zero, SafetyReport(SafetyState.ESTOP,
                                    ["emergency stop active: " + self.estop_reason],
                                    velocity_scale=0.0)

        # 1) The current state must already be safe, otherwise e-stop.
        current = self.check_configuration(q)
        if current.violations:
            self.trigger_estop("; ".join(current.violations), t)
            return zero, SafetyReport(SafetyState.ESTOP, current.violations,
                                    current.min_clearance, 0.0)
        self.last_safe_q = q.copy()

        scale = 1.0

        # 2) Joint velocity limit (uniform scaling keeps the direction).
        peak = float(np.max(np.abs(qdot)))
        if peak > self.cfg.qdot_max:
            f = self.cfg.qdot_max / peak
            qdot *= f
            scale *= f

        # 3) TCP linear speed limit.
        v_tip = self._tcp_linear_velocity(q, qdot)
        speed = float(np.linalg.norm(v_tip))
        if speed > self.cfg.tcp_speed_max:
            f = self.cfg.tcp_speed_max / speed
            qdot *= f
            scale *= f

        # 4) Predict the next configuration and refuse unsafe motion.
        q_next = q + qdot * dt
        predicted = self.check_configuration(q_next)
        if predicted.violations:
            self._log(t, predicted.violations)
            return zero, SafetyReport(SafetyState.BLOCKED, predicted.violations,
                                    predicted.min_clearance, 0.0)

        # 5) Slow down smoothly when close to a zone or boundary.
        state = SafetyState.OK
        if predicted.min_clearance < self.cfg.warn_distance:
            f = max(predicted.min_clearance / self.cfg.warn_distance, 0.1)
            qdot *= f
            scale *= f
            state = SafetyState.WARNING

        return qdot, SafetyReport(state, [], predicted.min_clearance, scale)

    def _tcp_linear_velocity(self, q, qdot) -> np.ndarray:
        """Linear velocity of the instrument tip (includes tool length)."""
        twist = jacobian_base(self.robot, q) @ qdot
        v, w = twist[:3], twist[3:]
        R = self.robot.fkine(q).R
        r = R[:, 2] * self.cfg.tool_length
        return v + np.cross(w, r)

    # ------------------------------------------------------------------
    # Emergency stop and recovery
    # ------------------------------------------------------------------
    def trigger_estop(self, reason: str = "manual", t: float = 0.0):
        self.estopped = True
        self.estop_reason = reason
        self._log(t, [f"EMERGENCY STOP: {reason}"])

    def reset_estop(self, q) -> bool:
        """Release the e-stop only if the current configuration is safe."""
        if self.check_configuration(q).violations:
            return False
        self.estopped = False
        self.estop_reason = ""
        return True

    def recovery_step(self, q, dt: float, speed_scale: float = 0.3):
        """Slow joint velocity moving back toward the last known safe pose.

        Returns (qdot, done). `done` is True when the robot has arrived.
        Returns (None, False) if no safe pose has been recorded yet.
        """
        if self.last_safe_q is None:
            return None, False
        q = np.asarray(q, dtype=float)
        err = self.last_safe_q - q
        if np.linalg.norm(err) < 1e-3:
            return np.zeros_like(q), True
        qdot = err / dt
        limit = self.cfg.qdot_max * speed_scale
        peak = float(np.max(np.abs(qdot)))
        if peak > limit:
            qdot *= limit / peak
        return qdot, False

    # ------------------------------------------------------------------
    def _log(self, t: float, messages):
        for m in messages:
            self.violations.append((t, m))

    @property
    def violation_count(self) -> int:
        return len(self.violations)