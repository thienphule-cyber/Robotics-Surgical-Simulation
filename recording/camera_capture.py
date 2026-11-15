"""Camera capture and video writing.

CameraCapture
    Holds named frame sources (functions that return a BGR image).
VideoStream
    Writes the frames of ONE camera to a video file and keeps an index file
    with the timestamps of every frame:
        <name>_frames.csv: frame, sample, t_sim, t_wall, image_file
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraCapture:
    def __init__(self, sources: dict | None = None):
        self.sources = dict(sources or {})
        self._failed = set()

    def add_source(self, name: str, grab: Callable[[], np.ndarray]):
        self.sources[name] = grab

    @property
    def names(self) -> list:
        return list(self.sources)

    def grab_all(self) -> dict:
        """Grab one frame from every source. Failing sources are skipped
        (the error is logged once per source)."""
        frames = {}
        for name, grab in self.sources.items():
            try:
                frame = grab()
            except Exception:
                if name not in self._failed:
                    self._failed.add(name)
                    log.exception("Camera source '%s' failed", name)
                continue
            if frame is not None:
                frames[name] = frame
        return frames


def eye_in_hand_source(workcell, overlay=None) -> Callable[[], np.ndarray]:
    """Frame source for the eye-in-hand camera of a Workcell.

    overlay : optional function returning (current_uv, desired_uv) or None;
        when given, the features are drawn on the image (like the camera panel).
    """
    from control import draw_overlay, forward_kinematics

    def grab():
        cam = workcell.eye_camera
        T_world_cam = cam.world_pose(forward_kinematics(workcell.robot, workcell.robot.q))
        image = cam.render(T_world_cam, workcell.marker_points)
        if overlay is not None:
            pair = overlay()
            if pair is not None:
                image = draw_overlay(image, pair[0], pair[1])
        return image

    return grab


def external_source(workcell) -> Callable[[], np.ndarray]:
    """Frame source for the external camera (wireframe view of the workcell)."""
    return lambda: workcell.render_external()


class VideoStream:
    # (fourcc, file suffix) tried in this order after the requested codec
    FALLBACK_CODECS = (("mp4v", ".mp4"), ("MJPG", ".avi"))

    def __init__(self, base_path, name: str, fps: float, frames_dir=None,
                save_every: int = 0, frame_ext: str = "jpg", codec: str | None = None):
        """
        base_path : video path without suffix (the suffix depends on the codec).
        save_every : also save every N-th frame as an image (0 = never).
        """
        self.base_path = Path(base_path)
        self.name = name
        self.fps = float(fps)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.save_every = int(save_every)
        self.frame_ext = frame_ext.lstrip(".").lower()
        self.codec = codec

        self.video_path = None
        self.size = None            # (width, height) of the first frame
        self.frame_count = 0
        self._writer = None

        self.index_path = self.base_path.parent / f"{name}_frames.csv"
        self._index_fh = open(self.index_path, "w", newline="")
        self._index = csv.writer(self._index_fh)
        self._index.writerow(["frame", "sample", "t_sim", "t_wall", "image_file"])

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(frame) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(frame)

    def _open_writer(self, width: int, height: int):
        candidates = []
        if self.codec:
            candidates.append((self.codec, ".avi" if self.codec.upper() == "MJPG" else ".mp4"))
        candidates.extend(self.FALLBACK_CODECS)

        for fourcc_name, suffix in candidates:
            path = Path(str(self.base_path) + suffix)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name),
                                    self.fps, (width, height))
            if writer.isOpened():
                self._writer = writer
                self.video_path = path
                return
            writer.release()
            path.unlink(missing_ok=True)
        raise RuntimeError("No usable video codec found (tried "
                        + ", ".join(c for c, _ in candidates) + ")")

    # ------------------------------------------------------------------
    def write(self, frame, sample: int, t_sim: float, t_wall: float):
        frame = self._normalize(frame)
        if self._writer is None:
            height, width = frame.shape[:2]
            self.size = (width, height)
            self._open_writer(width, height)
        elif (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)

        self._writer.write(frame)

        image_file = ""
        if self.save_every > 0 and self.frames_dir is not None \
                and self.frame_count % self.save_every == 0:
            filename = f"{self.name}_{self.frame_count:06d}.{self.frame_ext}"
            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if self.frame_ext in ("jpg", "jpeg") else []
            if cv2.imwrite(str(self.frames_dir / filename), frame, params):
                image_file = filename

        self._index.writerow([self.frame_count, sample, f"{t_sim:.6f}", f"{t_wall:.6f}", image_file])
        self.frame_count += 1
        if self.frame_count % 50 == 0:
            self._index_fh.flush()

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if not self._index_fh.closed:
            self._index_fh.close()

    def info(self) -> dict:
        return {
            "video": self.video_path.name if self.video_path else None,
            "index": self.index_path.name,
            "fps": self.fps,
            "resolution": list(self.size) if self.size else None,
            "frames": self.frame_count,
        }