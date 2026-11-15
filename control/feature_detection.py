"""Feature detection on camera images.

Two detectors are provided:
* ColorMarkerDetector: finds colour-coded circular markers. Every marker has
    its own hue, so each detection carries a fixed id (no correspondence issue).
* BlobDetector: finds bright blobs without ids (use with the tracker).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV hue values (0-179): red, yellow, green, cyan, blue, magenta.
MARKER_HUES = [0, 30, 60, 90, 120, 150]


def marker_bgr(index: int) -> tuple:
    """BGR colour used to draw / identify marker number `index`."""
    hue = MARKER_HUES[index % len(MARKER_HUES)]
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


@dataclass
class Detection:
    id: int | None      # marker id, or None for unlabeled detections
    uv: np.ndarray      # (2,) pixel coordinates (u, v)
    area: float         # blob area in pixels


class ColorMarkerDetector:
    def __init__(self, n_markers: int = 4, hue_tol: int = 8, s_min: int = 120,
                v_min: int = 80, min_area: float = 4.0):
        if n_markers > len(MARKER_HUES):
            raise ValueError(f"At most {len(MARKER_HUES)} markers are supported")
        self.n_markers = n_markers
        self.hue_tol = hue_tol
        self.s_min = s_min
        self.v_min = v_min
        self.min_area = min_area

    def detect(self, image_bgr: np.ndarray) -> list:
        """Return one Detection per marker colour found in the image."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.int16)
        bright = (hsv[:, :, 1] >= self.s_min) & (hsv[:, :, 2] >= self.v_min)

        detections = []
        for idx in range(self.n_markers):
            diff = np.abs(hue - MARKER_HUES[idx])
            diff = np.minimum(diff, 180 - diff)          # hue is circular
            mask = ((diff <= self.hue_tol) & bright).astype(np.uint8) * 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            uv = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            detections.append(Detection(idx, uv, float(area)))
        return detections


class BlobDetector:
    """Unlabeled bright-blob detector (threshold + contours)."""

    def __init__(self, threshold: int = 100, min_area: float = 4.0,
                max_blobs: int = 20):
        self.threshold = threshold
        self.min_area = min_area
        self.max_blobs = max_blobs

    def detect(self, image_bgr: np.ndarray) -> list:
        value = image_bgr.max(axis=2)                    # brightness (V channel)
        _, mask = cv2.threshold(value, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:self.max_blobs]:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            uv = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            detections.append(Detection(None, uv, float(area)))
        return detections


def draw_overlay(image_bgr: np.ndarray, current_uv=None, desired_uv=None) -> np.ndarray:
    """Return a copy of the image with the current (circles) and desired
    (crosses) feature positions drawn, plus a line between each pair.

    Intended for the Swift camera panel / recorded video.
    """
    out = image_bgr.copy()
    cur = None if current_uv is None else np.asarray(current_uv, dtype=float)
    des = None if desired_uv is None else np.asarray(desired_uv, dtype=float)

    if des is not None:
        for i, p in enumerate(des):
            if np.all(np.isfinite(p)):
                cv2.drawMarker(out, (int(round(p[0])), int(round(p[1]))),
                            (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    if cur is not None:
        for i, p in enumerate(cur):
            if not np.all(np.isfinite(p)):
                continue
            pt = (int(round(p[0])), int(round(p[1])))
            cv2.circle(out, pt, 10, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(out, str(i), (pt[0] + 12, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            if des is not None and i < len(des) and np.all(np.isfinite(des[i])):
                cv2.line(out, pt, (int(round(des[i][0])), int(round(des[i][1]))),
                        (255, 255, 0), 1, cv2.LINE_AA)
    return out