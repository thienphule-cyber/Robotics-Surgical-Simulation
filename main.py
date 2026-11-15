#!/usr/bin/env python3
"""main.py - UR3e vision-guided robotic surgical simulation (IBVS).

Usage
-----
python main.py                         Swift + dashboard, press Start in the browser
python main.py --auto-start --record   run immediately and record the session
python main.py --headless --once       no GUI, run one procedure and exit (fast)
python main.py --check                 check that the workcell layout is reachable

Procedure: initializing -> planning -> vision_init -> positioning ->
servoing -> evaluation -> completed (or aborted).
Every motion goes through MotionController, so the SafetyMonitor can always
override the commands. Dashboard commands are polled at every control step.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from spatialmath import SE3

from control import (
    ColorMarkerDetector,
    FeatureTracker,
    IBVSController,
    MotionController,
    SafetyMonitor,
    cartesian_to_joint_trajectory,
    cartesian_trajectory,
    desired_from_pose,
    draw_overlay,
    forward_kinematics,
    inverse_kinematics,
    pose_error,
    run_ibvs,
)
from dashboard import SystemState, start_dashboard
from recording import (
    CameraCapture,
    Recorder,
    build_row,
    eye_in_hand_source,
    external_source,
)
from workcell import Workcell, WorkcellConfig


class AbortRun(Exception):
    """Raised when the operator stops the run."""


class ProcedureError(Exception):
    """Raised when a phase of the procedure fails."""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="UR3e vision-guided surgical simulation (IBVS)")
    p.add_argument("--region", default="abdomen", help="surgical region (default: abdomen)")
    p.add_argument("--dt", type=float, default=0.02, help="control period in seconds")
    p.add_argument("--max-steps", type=int, default=1000, help="maximum IBVS steps")
    p.add_argument("--approach-time", type=float, default=3.0, help="duration of the approach motion (s)")
    p.add_argument("--tool-yaw", type=float, default=0.0, help="tool yaw about the vertical axis (deg)")
    p.add_argument("--depth", choices=["true", "desired"], default="true",
                help="IBVS depth: ground truth from the simulation, or the desired depth only")
    p.add_argument("--noise", type=float, default=0.0, help="camera image noise (grey levels)")
    p.add_argument("--headless", action="store_true", help="do not start Swift")
    p.add_argument("--no-dashboard", action="store_true", help="do not start the dashboard")
    p.add_argument("--http-port", type=int, default=5000)
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--auto-start", action="store_true", help="start the procedure immediately")
    p.add_argument("--once", action="store_true", help="run one procedure, then exit (implies --auto-start)")
    p.add_argument("--record", action="store_true", help="record every procedure automatically")
    p.add_argument("--record-dir", default="recordings")
    p.add_argument("--show-camera", action="store_true", help="show the eye-in-hand camera in an OpenCV window")
    p.add_argument("--pace", action="store_true", help="run in real time (always on with Swift)")
    p.add_argument("--check", action="store_true", help="check reachability of the layout and exit")
    args = p.parse_args(argv)
    if args.once:
        args.auto_start = True
    return args


class Application:
    def __init__(self, args):
        self.args = args
        self.log = logging.getLogger("main")

        self.wc = None
        self.state = None
        self.server = None
        self.monitor = None
        self.ctrl = None
        self.ibvs = None
        self.detector = None
        self.tracker = None
        self.rec = None

        self.running = False
        self.paused = False
        self.exit_code = 0
        self.last_result = None      # latest IBVSResult
        self.last_uv = None          # latest tracked feature positions
        self.T_goal = None

        self._in_poll = False
        self._cfg_version = -1
        self._fps = 0.0
        self._t_frame = time.perf_counter()
        self._tick = None
        self._t0_sim = 0.0
        self._t0_wall = time.perf_counter()

    # ------------------------------------------------------------------
    # Setup and shutdown
    # ------------------------------------------------------------------
    def info(self, message: str, level: str = "info"):
        getattr(self.log, level)(message)
        self.state.log(message, level=level, source="main")

    def setup(self):
        a = self.args
        cfg = WorkcellConfig(default_region=a.region, tool_yaw_deg=a.tool_yaw)
        if a.region not in cfg.regions:
            raise SystemExit(f"Unknown region '{a.region}'. Available: {list(cfg.regions)}")

        self.wc = Workcell(cfg, headless=a.headless)
        self.wc.eye_camera.noise_std = a.noise

        self.state = SystemState(regions=list(cfg.regions), default_region=self.wc.region)
        self.state.set_region(self.wc.region, self.wc.site)
        conf = self.state.get_config()
        self._cfg_version = self.state.config_version

        self.monitor = SafetyMonitor(self.wc.robot, self.wc.make_safety_config(
            qdot_max=conf["qdot_max"], tcp_speed_max=conf["tcp_speed_max"],
            warn_distance=conf["warn_distance"]))
        self.ctrl = MotionController(self.wc.robot, dt=a.dt, qdot_max=conf["qdot_max"],
                                    safety=self.monitor, on_step=self.on_step)

        self.ibvs = IBVSController(
            self.wc.eye_camera.intrinsics, gain=conf["ibvs_gain"], tol_px=conf["ibvs_tol_px"],
            depth_mode="mean" if a.depth == "true" else "desired")
        self.detector = ColorMarkerDetector(n_markers=cfg.n_markers)
        self.tracker = FeatureTracker()

        capture = CameraCapture({
            "eye_in_hand": eye_in_hand_source(self.wc, overlay=self._overlay),
            "external": external_source(self.wc),
        })
        self.rec = Recorder(root=a.record_dir, video_fps=25.0, capture=capture)

        if not a.no_dashboard:
            self.server = start_dashboard(self.state, http_port=a.http_port, ws_port=a.ws_port)
            self.info(f"Dashboard: {self.server.url}")

    def shutdown(self):
        try:
            if self.rec is not None:
                self.rec.stop()
        except Exception:
            self.log.exception("Could not close the recording")
        if self.server is not None:
            self.server.stop()
        if self.wc is not None:
            self.wc.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    # ------------------------------------------------------------------
    # Per-step callback (runs for EVERY control step of every phase)
    # ------------------------------------------------------------------
    def on_step(self, c):
        self.wc.step(c.dt)
        ms = c.get_state()
        self.state.update_from_motion(ms)
        self.state.update_from_safety_monitor(self.monitor, c.last_report)
        if self.rec.is_recording:
            self.rec.record(c.t, self._row(ms, c))
        if len(c.history) > 5000:
            del c.history[:2500]
        self._pace()
        self.poll_commands()
        while self.paused:                       # blocks the run while paused
            time.sleep(0.02)
            self.wc.step(0.02)
            self.poll_commands()

    def _pace(self):
        if not (self.args.pace or not self.args.headless):
            return
        now = time.perf_counter()
        if self._tick is None:
            self._tick = now
        self._tick += self.args.dt
        delay = self._tick - now
        if delay > 0:
            time.sleep(delay)
        elif delay < -1.0:
            self._tick = now                     # fell behind: resynchronise

    def _row(self, ms, c) -> dict:
        qdot = c.history[-1]["qdot"] if c.history else None
        row = build_row(ms, ibvs_result=self.last_result, qdot=qdot, phase=self.state.phase)
        if self.last_result is None:             # keep the CSV columns complete from row 1
            row.update({"ibvs_error_px": float("nan"), "ibvs_status": "",
                        "ibvs_n_features": 0, "ibvs_converged": False})
            row.update({f"twist_{k}": 0.0 for k in ("vx", "vy", "vz", "wx", "wy", "wz")})
        return row

    def _overlay(self):
        if self.last_uv is None or self.ibvs.desired_uv is None:
            return None
        return self.last_uv, self.ibvs.desired_uv

    def on_frame(self, image, uv, desired_uv, result):
        """Called by run_ibvs at every servoing step."""
        self.last_uv, self.last_result = uv, result
        now = time.perf_counter()
        dt = now - self._t_frame
        self._t_frame = now
        if dt > 0:
            rate = 1.0 / dt
            self._fps = rate if self._fps == 0.0 else 0.9 * self._fps + 0.1 * rate
        self.state.update_from_ibvs(result, uv, desired_uv)
        self.state.update_vision(fps=self._fps)
        if self.args.show_camera:
            cv2.imshow("eye-in-hand camera", draw_overlay(image, uv, desired_uv))
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Dashboard commands
    # ------------------------------------------------------------------
    def apply_config(self):
        """Apply parameters changed from the dashboard."""
        if self.state.config_version == self._cfg_version:
            return
        c = self.state.get_config()
        self._cfg_version = self.state.config_version
        self.ibvs.gain = c["ibvs_gain"]
        self.ibvs.tol_px = c["ibvs_tol_px"]
        self.monitor.cfg.tcp_speed_max = c["tcp_speed_max"]
        self.monitor.cfg.qdot_max = c["qdot_max"]
        self.monitor.cfg.warn_distance = c["warn_distance"]
        self.ctrl.qdot_max = c["qdot_max"]

    def poll_commands(self, idle: bool = False) -> bool:
        """Process queued commands. Returns True if 'start' was requested
        while idle. Raises AbortRun if the operator stops a running procedure."""
        if self._in_poll:
            return False
        self._in_poll = True
        try:
            self.apply_config()
            start = False
            for cmd in self.state.pop_commands():
                name, payload = cmd["command"], cmd.get("payload", {})
                if name == "start":
                    if idle:
                        start = True
                    else:
                        self.info("Start ignored: a run is already in progress", "warning")
                elif name == "pause":
                    self._set_pause(True)
                elif name == "resume":
                    self._set_pause(False)
                elif name == "stop":
                    if self.running:
                        raise AbortRun("stopped by the operator")
                    self.info("Stop ignored: nothing is running", "warning")
                elif name == "estop":
                    self.monitor.trigger_estop("triggered from the dashboard", self.ctrl.t)
                    self.state.update_safety("estop", True, self.monitor.violation_count,
                                            None, ["triggered from the dashboard"])
                elif name == "reset_estop":
                    self._reset_estop()
                elif name == "reset":
                    if not self.running:
                        self.state.reset_procedure()
                        self.info("Procedure reset")
                elif name == "start_recording":
                    self.start_recording()
                elif name == "stop_recording":
                    self.stop_recording()
                elif name == "select_region":
                    self._select_region(payload.get("region"))
            return start
        finally:
            self._in_poll = False

    def _set_pause(self, flag: bool):
        if not self.running:
            self.info("Pause/resume ignored: no run in progress", "warning")
            return
        if self.paused == flag:
            return
        self.paused = flag
        self.state.set_flags(paused=flag)
        if flag:
            self.rec.pause()
        else:
            self.rec.resume()
        self.info("Run paused" if flag else "Run resumed")

    def _reset_estop(self):
        if not self.monitor.estopped:
            return
        if self.monitor.reset_estop(self.ctrl.q):
            ok = True
        else:
            ok = self.ctrl.recover()          # move back to the last safe pose
        self.state.update_from_safety_monitor(self.monitor, None)
        self.info("Emergency stop released" if ok else "Could not release the emergency stop",
                "info" if ok else "error")

    def _select_region(self, region):
        if self.running or region not in self.wc.cfg.regions:
            self.info(f"Region change to '{region}' ignored", "warning")
            self.state.set_region(self.wc.region, self.wc.site)   # undo the dashboard's change
            return
        self.wc.select_region(region)
        self.state.set_region(region, self.wc.site)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def start_recording(self):
        if self.rec.is_recording:
            return
        name = f"{self.wc.region}_{datetime.now():%Y%m%d_%H%M%S}"
        path = self.rec.start(name=name, metadata={
            "region": self.wc.region,
            "site": self.wc.site,
            "marker_points": self.wc.marker_points,
            "desired_flange_pose": self.wc.desired_flange_pose().A,
            "config": self.state.get_config(),
            "dt": self.args.dt,
        })
        self.state.set_flags(recording=True)
        self.info(f"Recording to {path}")

    def stop_recording(self):
        path = self.rec.stop()
        self.state.set_flags(recording=False)
        if path:
            self.info(f"Recording saved: {path}")

    # ------------------------------------------------------------------
    # Procedure
    # ------------------------------------------------------------------
    def _ready_pose(self) -> SE3:
        return SE3.Trans(-0.04, 0.0, 0.04) * self.wc.start_flange_pose()   # was Trans(0, 0, 0.10)

    def _last_violation(self) -> str:
        return self.monitor.violations[-1][1] if self.monitor.violations else "unknown reason"

    def run_procedure(self) -> str:
        """Run one complete procedure. Returns 'converged' on success."""
        if self.monitor.estopped:
            self.info("Cannot start: the emergency stop is active. Reset it first.", "error")
            return "estop_active"

        self.running = True
        self.paused = False
        self._tick = None
        self._t0_sim = self.ctrl.t
        self._t0_wall = time.perf_counter()
        self.state.reset_procedure()
        self.state.set_flags(running=True, paused=False)

        auto_recording = False
        result = "failed"
        try:
            if self.args.record and not self.rec.is_recording:
                self.start_recording()
                auto_recording = True
            self._phase_initialize()
            traj = self._phase_plan()
            self._phase_vision_init()
            self._phase_position(traj)
            status, log = self._phase_servo()
            self._phase_evaluate(status, log)
            result = status
        except AbortRun as exc:
            self.state.set_phase("aborted", str(exc))
            result = "stopped"
        except ProcedureError as exc:
            self.state.set_phase("aborted", str(exc))
            self.info(str(exc), "error")
            result = "failed"
        finally:
            self.running = False
            self.paused = False
            self.state.set_flags(running=False, paused=False)
            if self.rec.is_recording:
                self.rec.session.update_metadata(procedure_status=result)
            if auto_recording:
                self.stop_recording()
        return result

    def _phase_initialize(self):
        self.state.set_phase("initializing", "placing the robot at the ready pose")
        self.tracker.reset()
        self.last_result = None
        self.last_uv = None
        self.ctrl.last_report = None
        self.T_goal = self.wc.desired_flange_pose()

        q, ok = inverse_kinematics(self.wc.robot, self._ready_pose(), q0=self.wc.robot.q)
        if not ok:
            raise ProcedureError("Ready pose is unreachable. Run 'python main.py --check' "
                                "and adjust the workcell layout")
        report = self.monitor.check_configuration(q)
        if report.violations:
            raise ProcedureError("Ready pose violates safety: " + "; ".join(report.violations))
        self.wc.robot.q = q                      # teleport
        self.wc.step(0.01)
        self.monitor.last_safe_q = q.copy()
        self.state.update_from_motion(self.ctrl.get_state())

    def _phase_plan(self):
        self.state.set_phase("planning", f"region '{self.wc.region}'")
        robot = self.wc.robot
        T_now = forward_kinematics(robot, robot.q)
        ctraj = cartesian_trajectory(T_now, self.wc.start_flange_pose(),
                                    self.args.approach_time, self.args.dt)
        try:
            traj = cartesian_to_joint_trajectory(robot, ctraj, q0=robot.q)
        except ValueError as exc:
            raise ProcedureError(f"Approach path is not reachable: {exc}")

        for i, q in enumerate(traj.q):           # the whole path must be safe
            violations = self.monitor.check_configuration(q).violations
            if violations:
                raise ProcedureError(f"Approach path unsafe at waypoint {i}: " + "; ".join(violations))

        q_goal, ok = inverse_kinematics(robot, self.T_goal, q0=traj.q[-1])
        if not ok:
            raise ProcedureError("The goal pose is unreachable")
        violations = self.monitor.check_configuration(q_goal).violations
        if violations:
            raise ProcedureError("The goal pose violates safety: " + "; ".join(violations))
        return traj

    def _phase_vision_init(self):
        self.state.set_phase("vision_init", "detecting the target markers")
        cam = self.wc.eye_camera
        points = self.wc.marker_points
        image = cam.render(cam.world_pose(forward_kinematics(self.wc.robot, self.wc.robot.q)), points)
        detections = sorted(self.detector.detect(image), key=lambda d: d.id)
        if len(detections) < len(points):
            raise ProcedureError(f"Only {len(detections)}/{len(points)} markers are visible "
                                "from the ready pose")
        try:
            desired_uv, desired_depth = desired_from_pose(
                cam, cam.world_pose(self.T_goal), points)
        except ValueError as exc:
            raise ProcedureError(str(exc))

        self.ibvs.set_desired(desired_uv, desired_depth)
        uv = np.array([d.uv for d in detections])
        self.last_uv = uv
        self.state.update_vision(uv=uv, desired_uv=desired_uv, n_features=len(detections),
                                status="markers detected", camera=cam.name)

    def _phase_position(self, traj):
        self.state.set_phase("positioning", "moving to the approach pose")
        if not self.ctrl.follow_joint_trajectory(traj):
            raise ProcedureError("Approach motion stopped: " + self._last_violation())

    def _phase_servo(self):
        self.state.set_phase("servoing", "visual servoing to the target")
        self.tracker.reset()
        self._fps = 0.0
        self._t_frame = time.perf_counter()
        return run_ibvs(self.ctrl, self.wc.eye_camera, self.detector, self.tracker, self.ibvs,
                        self.wc.marker_points, max_steps=self.args.max_steps,
                        use_true_depth=(self.args.depth == "true"), on_frame=self.on_frame)

    def _phase_evaluate(self, status: str, log: list):
        self.state.set_phase("evaluation", "computing metrics")
        self.state.update_performance(**self._compute_metrics(status, log))
        if status == "converged":
            self.state.set_phase("completed", "target reached")
            self.info("Procedure completed")
            return
        reasons = {
            "blocked": "motion stopped by safety: " + self._last_violation(),
            "lost_features": "too few features were visible",
            "max_steps": f"no convergence in {self.args.max_steps} steps",
        }
        raise ProcedureError("Visual servoing ended: " + reasons.get(status, status))

    def _compute_metrics(self, status: str, log: list) -> dict:
        robot = self.wc.robot
        err = pose_error(forward_kinematics(robot, robot.q), self.T_goal)
        px = np.array([e["error_px"] for e in log], dtype=float)
        finite = px[np.isfinite(px)]
        wall = time.perf_counter() - self._t0_wall
        sim = self.ctrl.t - self._t0_sim
        metrics = {
            "status": status,
            "region": self.wc.region,
            "servo_steps": len(log),
            "initial_ibvs_error_px": float(finite[0]) if finite.size else None,
            "final_ibvs_error_px": float(finite[-1]) if finite.size else None,
            "convergence_time_s": (log[-1]["t"] - log[0]["t"]) if (log and status == "converged") else None,
            "position_error_mm": float(np.linalg.norm(err[:3]) * 1000.0),
            "orientation_error_deg": float(np.degrees(np.linalg.norm(err[3:]))),
            "execution_time_sim_s": float(sim),
            "execution_time_wall_s": float(wall),
            "avg_loop_rate_hz": float(sim / self.args.dt / wall) if wall > 0 else None,
            "safety_violations": self.monitor.violation_count,
        }
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Layout check
    # ------------------------------------------------------------------
    def check(self) -> int:
        """Check IK reachability, safety and marker visibility. Returns an exit code."""
        wc = self.wc
        default_region = wc.region
        cam = wc.eye_camera
        problems = 0
        print(f"{'region':<10} {'pose':<8} result")
        for region in wc.cfg.regions:
            wc.select_region(region)
            poses = {"goal": wc.desired_flange_pose(), "start": wc.start_flange_pose(),
                    "ready": self._ready_pose()}
            for name, pose in poses.items():
                q, ok = inverse_kinematics(wc.robot, pose, q0=wc.robot.q)
                if not ok:
                    verdict = "IK FAILED (unreachable)"
                else:
                    violations = self.monitor.check_configuration(q).violations
                    verdict = "ok" if not violations else "UNSAFE: " + "; ".join(violations)
                problems += verdict != "ok"
                print(f"{region:<10} {name:<8} {verdict}")
            try:
                desired_from_pose(cam, cam.world_pose(poses["goal"]), wc.marker_points)
                print(f"{region:<10} {'markers':<8} visible from the goal pose")
            except ValueError as exc:
                problems += 1
                print(f"{region:<10} {'markers':<8} {exc}")
        wc.select_region(default_region)
        print("Layout OK" if problems == 0 else f"{problems} problem(s) found: adjust WorkcellConfig")
        return 0 if problems == 0 else 1

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        a = self.args
        auto = a.auto_start
        if a.no_dashboard and not auto:
            print("No dashboard: starting the procedure immediately")
            auto = True

        self.info("System ready" + ("" if auto else ": press Start in the dashboard"))
        if auto:
            result = self.run_procedure()
            self.exit_code = 0 if result == "converged" else 1
            if a.once or a.no_dashboard:
                return self.exit_code

        while True:
            if self.poll_commands(idle=True):
                result = self.run_procedure()
                self.exit_code = 0 if result == "converged" else 1
                if a.once:
                    return self.exit_code
            self.wc.step(0.05)
            time.sleep(0.02)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Application(args)
    try:
        app.setup()
        if args.check:
            return app.check()
        return app.run()
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    finally:
        app.shutdown()


if __name__ == "__main__":
    sys.exit(main())