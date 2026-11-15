"""Motion controller: the single place where commands reach the robot.

Pipeline for every control step:
    command (qdot or twist)
    -> joint velocity limiting
    -> SafetyMonitor.filter_velocity   (safety can override anything)
      -> integrate q <- q + qdot * dt
    -> write robot.q and call on_step (e.g. to step the Swift simulator)
"""

from __future__ import annotations

import numpy as np

from .jacobian import (
    cartesian_to_joint_velocity,
    jacobian_base,
    jacobian_camera,
    jacobian_tool,
    limit_joint_velocity,
)
from .kinematics import clip_to_joint_limits, forward_kinematics
from .safety import SafetyMonitor, SafetyState
from .trajectory import (
    cartesian_to_joint_trajectory,
    cartesian_trajectory,
    joint_trajectory,
)


class MotionController:
    def __init__(self, robot, dt: float = 0.02, qdot_max=1.0, kp: float = 5.0,
                safety: SafetyMonitor | None = None, on_step=None,
                keep_history: bool = True):
        """
        Parameters
        ----------
        robot : roboticstoolbox robot (see kinematics.build_ur3e).
        dt : control period [s].
        qdot_max : joint speed limit [rad/s] (scalar or length-n array).
        kp : proportional gain for trajectory tracking.
        safety : optional SafetyMonitor (strongly recommended).
        on_step : callable(controller) invoked after every step, for example
                lambda c: env.step(c.dt) to advance the Swift simulation.
        """
        self.robot = robot
        self.dt = dt
        self.qdot_max = qdot_max
        self.kp = kp
        self.safety = safety
        self.on_step = on_step
        self.keep_history = keep_history
        self.t = 0.0
        self.history = []        # list of dicts for recording/evaluation
        self.last_report = None

    # ------------------------------------------------------------------
    @property
    def q(self) -> np.ndarray:
        return np.asarray(self.robot.q, dtype=float).copy()

    def _apply(self, qdot):
        """Filter, integrate and apply one joint velocity command."""
        qdot = limit_joint_velocity(np.asarray(qdot, dtype=float), self.qdot_max)

        report = None
        if self.safety is not None:
            qdot, report = self.safety.filter_velocity(self.q, qdot, self.dt, self.t)

        self.robot.q = clip_to_joint_limits(self.robot, self.q + qdot * self.dt)
        self.t += self.dt
        self.last_report = report

        if self.keep_history:
            self.history.append({
                "t": self.t,
                "q": self.q,
                "qdot": np.asarray(qdot, dtype=float).copy(),
                "state": report.state.value if report else "unmonitored",
            })
        if self.on_step is not None:
            self.on_step(self)
        return report

    # ------------------------------------------------------------------
    # Low-level commands (used directly by IBVS)
    # ------------------------------------------------------------------
    def command_joint_velocity(self, qdot):
        return self._apply(qdot)

    def command_twist(self, twist, frame: str = "base", T_tool_cam=None,
                    damping: float = 0.05):
        """Command a Cartesian twist [v; w] expressed in the given frame.

        frame: "base", "tool" or "camera" (camera needs T_tool_cam, the pose
        of the eye-in-hand camera in the tool frame). IBVS uses "camera".
        """
        if frame == "base":
            J = jacobian_base(self.robot, self.q)
        elif frame == "tool":
            J = jacobian_tool(self.robot, self.q)
        elif frame == "camera":
            if T_tool_cam is None:
                raise ValueError("T_tool_cam is required for the camera frame")
            J = jacobian_camera(self.robot, self.q, T_tool_cam)
        else:
            raise ValueError("frame must be 'base', 'tool' or 'camera'")

        qdot = cartesian_to_joint_velocity(J, twist, damping=damping)
        return self._apply(qdot)

    def stop(self):
        """Hold position for one step."""
        return self._apply(np.zeros(self.robot.n))

    # ------------------------------------------------------------------
    # High-level motions
    # ------------------------------------------------------------------
    def follow_joint_trajectory(self, traj, kp: float | None = None) -> bool:
        """Track a joint trajectory (feed-forward + P feedback).

        Returns False if the safety monitor blocked the motion or an
        emergency stop occurred, True if the trajectory completed.
        """
        gain = self.kp if kp is None else kp
        for i in range(len(traj)):
            qdot = traj.qd[i] + gain * (traj.q[i] - self.q)
            report = self._apply(qdot)
            if report is not None and report.state in (SafetyState.BLOCKED,
                                                    SafetyState.ESTOP):
                return False
        return True

    def move_to_joint(self, q_goal, duration: float = 3.0,
                    profile: str = "quintic") -> bool:
        traj = joint_trajectory(self.q, q_goal, duration, self.dt, profile)
        return self.follow_joint_trajectory(traj)

    def move_to_pose(self, T_goal, duration: float = 3.0, profile: str = "quintic",
                    linear: bool = True) -> bool:
        """Move the tool to a pose.

        linear=True: straight line in Cartesian space (IK along the path,
        raises ValueError if IK fails). linear=False: IK once, then a
        joint-space move.
        """
        if linear:
            T_now = forward_kinematics(self.robot, self.q)
            ctraj = cartesian_trajectory(T_now, T_goal, duration, self.dt, profile)
            traj = cartesian_to_joint_trajectory(self.robot, ctraj, q0=self.q)
            return self.follow_joint_trajectory(traj)

        from .kinematics import inverse_kinematics
        q_goal, ok = inverse_kinematics(self.robot, T_goal, q0=self.q)
        if not ok:
            raise ValueError("IK failed for the requested pose")
        return self.move_to_joint(q_goal, duration, profile)

    # ------------------------------------------------------------------
    # Safe recovery after an emergency stop
    # ------------------------------------------------------------------
    def recover(self, max_steps: int = 500) -> bool:
        """Slowly return to the last safe pose, then release the e-stop.

        This bypasses the normal velocity filter on purpose (the current pose
        is unsafe), but only moves along the short path back to a pose that
        was verified safe.
        """
        if self.safety is None:
            return False
        for _ in range(max_steps):
            qdot, done = self.safety.recovery_step(self.q, self.dt)
            if qdot is None:
                return False
            if done:
                return self.safety.reset_estop(self.q)
            self.robot.q = clip_to_joint_limits(self.robot, self.q + qdot * self.dt)
            self.t += self.dt
            if self.on_step is not None:
                self.on_step(self)
        return False

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Snapshot for the dashboard / recorder."""
        T = forward_kinematics(self.robot, self.q)
        state = self.last_report.state.value if self.last_report else "unmonitored"
        return {
            "t": self.t,
            "q": self.q.tolist(),
            "tcp_position": np.asarray(T.t).tolist(),
            "safety_state": state,
            "estop": bool(self.safety.estopped) if self.safety else False,
        }