"""Recorder: start / pause / resume / stop recording of a simulation run.

Everything shares one clock (the simulation time t_sim) and a sample index:
  * robot_log.csv       one row per sample: sample, t_sim, t_rec, t_wall, ...
  * <camera>.mp4        video written at `video_fps` of SIMULATION time
  * <camera>_frames.csv timestamps of every video frame (same `sample`)
  * events.jsonl        start / pause / resume / stop and custom events
so any video frame can be matched with the robot state and IBVS error.
While paused nothing is recorded (the timestamps show the gap).

Single-threaded use: call everything from the main loop.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from enum import Enum

import cv2
import numpy as np

from .camera_capture import CameraCapture, VideoStream
from .session_manager import (
    EVENTS_FILE,
    ROBOT_LOG_FILE,
    SessionManager,
    json_default,
    sanitize_name,
)

log = logging.getLogger(__name__)


class RecorderState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


class _RowLog:
    """CSV writer with flexible columns (fixed by the first row)."""

    LEADING = ["sample", "t_sim", "t_rec", "t_wall"]

    def __init__(self, path, columns=None):
        self._fh = open(path, "w", newline="")
        self._columns = list(columns) if columns else None
        self._writer = None
        self._count = 0
        self._warned = False

    def write(self, sample, t_sim, t_rec, t_wall, row: dict):
        if self._writer is None:
            if self._columns is None:
                self._columns = [k for k in row if k not in self.LEADING]
            self._writer = csv.DictWriter(self._fh, fieldnames=self.LEADING + self._columns,
                                          extrasaction="ignore", restval="")
            self._writer.writeheader()
        elif not self._warned and (set(row) - set(self._columns) - set(self.LEADING)):
            self._warned = True
            log.warning("robot_log: new columns after the first row are ignored; "
                        "pass a complete first row or `row_columns` to Recorder")

        out = dict(row)
        out.update({"sample": sample, "t_sim": f"{t_sim:.6f}",
                    "t_rec": f"{t_rec:.6f}", "t_wall": f"{t_wall:.6f}"})
        self._writer.writerow(out)
        self._count += 1
        if self._count % 50 == 0:
            self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()


def _put_array(row: dict, prefix: str, values, names=None):
    if values is None:
        return
    for i, v in enumerate(np.asarray(values, dtype=float).ravel()):
        key = f"{prefix}_{names[i]}" if names else f"{prefix}{i + 1}"
        row[key] = float(v)


def build_row(motion_state: dict | None = None, ibvs_result=None, qdot=None,
              phase: str | None = None, extra: dict | None = None) -> dict:
    """Flatten the simulation state into one log row.

    motion_state : dict from MotionController.get_state()
    ibvs_result  : IBVSResult (error, status, twist...)
    The FIRST row decides the CSV columns, so give it all fields (for example
    start recording right before the IBVS loop begins).
    """
    row = {}
    if motion_state is not None:
        _put_array(row, "q", motion_state.get("q"))
        _put_array(row, "tcp", motion_state.get("tcp_position"), names=("x", "y", "z"))
        row["safety_state"] = motion_state.get("safety_state")
        row["estop"] = motion_state.get("estop")
    _put_array(row, "qdot", qdot)
    if ibvs_result is not None:
        row["ibvs_error_px"] = float(ibvs_result.error_px)
        row["ibvs_status"] = ibvs_result.status
        row["ibvs_n_features"] = int(ibvs_result.n_features)
        row["ibvs_converged"] = bool(ibvs_result.converged)
        _put_array(row, "twist", ibvs_result.twist, names=("vx", "vy", "vz", "wx", "wy", "wz"))
    if phase is not None:
        row["phase"] = phase
    if extra:
        row.update(extra)
    return row


class Recorder:
    def __init__(self, root="recordings", video_fps: float = 25.0, log_every: int = 1,
                 save_frames_every: int = 0, frame_ext: str = "jpg", codec: str | None = None,
                 capture: CameraCapture | None = None, row_columns=None):
        """
        video_fps : video frame rate in simulation seconds.
        log_every : write a robot_log row every N recorded samples.
        save_frames_every : also save every N-th video frame as an image (0 = off).
        capture : default frame sources, used when record() gets no frames.
        row_columns : fix the robot_log columns in advance (optional).
        """
        self.manager = SessionManager(root)
        self.video_fps = float(video_fps)
        self.log_every = max(1, int(log_every))
        self.save_frames_every = int(save_frames_every)
        self.frame_ext = frame_ext
        self.codec = codec
        self.capture = capture
        self.row_columns = row_columns

        self.state = RecorderState.IDLE
        self.session = None
        self._streams = {}
        self._log = None
        self._events = None
        self._sample = 0
        self._t0_sim = None
        self._t0_wall = None
        self._last_t = 0.0
        self._next_frame_t = None

    # ------------------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        return self.state is RecorderState.RECORDING

    def start(self, name: str | None = None, metadata: dict | None = None):
        """Start a new session. Returns the session folder."""
        if self.state in (RecorderState.RECORDING, RecorderState.PAUSED):
            raise RuntimeError("A recording is already in progress")

        self.session = self.manager.create(name, metadata)
        self._log = _RowLog(self.session.file(ROBOT_LOG_FILE), self.row_columns)
        self._events = open(self.session.file(EVENTS_FILE), "w")
        self._streams = {}
        self._sample = 0
        self._t0_sim = None
        self._t0_wall = None
        self._last_t = 0.0
        self._next_frame_t = None

        self.state = RecorderState.RECORDING
        self.session.update_metadata(state="recording", video_fps=self.video_fps,
                                     log_every=self.log_every,
                                     started_at=datetime.now().isoformat(timespec="seconds"))
        self.add_event(None, "start", "recording started")
        return self.session.path

    def pause(self) -> bool:
        if self.state is not RecorderState.RECORDING:
            return False
        self.state = RecorderState.PAUSED
        self.add_event(None, "pause", "recording paused")
        return True

    def resume(self) -> bool:
        if self.state is not RecorderState.PAUSED:
            return False
        self._next_frame_t = None          # write a frame right after resuming
        self.state = RecorderState.RECORDING
        self.add_event(None, "resume", "recording resumed")
        return True

    def stop(self):
        """Stop recording, close all files, write the metadata.
        Returns the session folder (None if nothing was recording)."""
        if self.state not in (RecorderState.RECORDING, RecorderState.PAUSED):
            return None
        self.add_event(None, "stop", "recording stopped")

        for stream in self._streams.values():
            stream.close()
        self._log.close()
        self._events.close()

        t_wall_end = time.time()
        self.session.update_metadata(
            state="stopped",
            ended_at=datetime.now().isoformat(timespec="seconds"),
            samples=self._sample,
            duration_sim_s=(self._last_t - self._t0_sim) if self._t0_sim is not None else 0.0,
            duration_wall_s=(t_wall_end - self._t0_wall) if self._t0_wall is not None else 0.0,
            cameras={name: s.info() for name, s in self._streams.items()},
            files={"robot_log": ROBOT_LOG_FILE, "events": EVENTS_FILE},
        )
        self.state = RecorderState.STOPPED
        return self.session.path

    # ------------------------------------------------------------------
    def _stream(self, name: str) -> VideoStream:
        if name not in self._streams:
            safe = sanitize_name(name)
            frames_dir = self.session.dir("frames", safe) if self.save_frames_every > 0 else None
            self._streams[name] = VideoStream(
                self.session.path / safe, name=safe, fps=self.video_fps,
                frames_dir=frames_dir, save_every=self.save_frames_every,
                frame_ext=self.frame_ext, codec=self.codec)
        return self._streams[name]

    def _frame_due(self, t: float) -> bool:
        if self._next_frame_t is None or t >= self._next_frame_t - 1e-9:
            period = 1.0 / self.video_fps
            nxt = (self._next_frame_t if self._next_frame_t is not None else t) + period
            self._next_frame_t = nxt if nxt > t else t + period
            return True
        return False

    def record(self, t: float, row: dict | None = None, frames=None) -> bool:
        """Record one control step. Returns True if the sample was stored.

        t      : simulation time (the master clock).
        row    : robot/control values (see build_row).
        frames : dict {camera name: BGR image}, or a function returning such a
                 dict (called only when a video frame is due, so rendering is
                 skipped otherwise). If None, `capture.grab_all()` is used.
        """
        if self.state is not RecorderState.RECORDING:
            return False

        t_wall = time.time()
        if self._t0_sim is None:
            self._t0_sim, self._t0_wall = t, t_wall
        sample = self._sample
        self._sample += 1
        self._last_t = t

        if row is not None and sample % self.log_every == 0:
            self._log.write(sample, t, t - self._t0_sim, t_wall, row)

        if self._frame_due(t):
            images = frames() if callable(frames) else frames
            if images is None and self.capture is not None:
                images = self.capture.grab_all()
            for name, image in (images or {}).items():
                self._stream(name).write(image, sample, t, t_wall)
        return True

    def add_event(self, t: float | None, kind: str, message: str = ""):
        """Add a line to events.jsonl (phase changes, safety events...)."""
        if self._events is None or self._events.closed:
            return
        entry = {"sample": self._sample,
                 "t_sim": self._last_t if t is None else t,
                 "t_wall": time.time(), "kind": kind, "message": message}
        self._events.write(json.dumps(entry, default=json_default) + "\n")
        self._events.flush()

    def save_snapshot(self) -> list:
        """Save one still image per camera (PNG) in the session's snapshots folder."""
        if self.session is None or self.capture is None or self.state not in (
                RecorderState.RECORDING, RecorderState.PAUSED):
            return []
        paths = []
        stamp = int(time.time() * 1000)
        for name, image in self.capture.grab_all().items():
            path = self.session.file("snapshots", f"{sanitize_name(name)}_{stamp}.png")
            if cv2.imwrite(str(path), image):
                paths.append(path)
        return paths

    def status(self) -> dict:
        """Summary for the dashboard."""
        return {
            "state": self.state.value,
            "session": self.session.name if self.session else None,
            "path": str(self.session.path) if self.session else None,
            "samples": self._sample,
            "frames": {name: s.frame_count for name, s in self._streams.items()},
            "elapsed_sim_s": (self._last_t - self._t0_sim) if self._t0_sim is not None else 0.0,
        }