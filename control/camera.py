"""Pinhole camera model and synthetic image rendering.

Swift does not render camera images, so the scene seen by the camera is
generated here: 3D marker points are projected with the pinhole model and
drawn with OpenCV. The feature detector then runs on the resulting image.

Camera frame convention (OpenCV): x right, y down, z forward (optical axis).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from spatialmath import SE3

from .feature_detection import marker_bgr
from .transformations import project_points, to_se3


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                        [0.0, self.fy, self.cy],
                        [0.0, 0.0, 1.0]])

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float = 60.0):
        """Build intrinsics from the horizontal field of view (square pixels)."""
        f = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, width, height)

    def pixel_to_normalized(self, uv) -> np.ndarray:
        """Pixels (N, 2) -> normalised image coordinates (x, y) = (X/Z, Y/Z)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return np.column_stack([(uv[:, 0] - self.cx) / self.fx,
                                (uv[:, 1] - self.cy) / self.fy])

    def normalized_to_pixel(self, xy) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        return np.column_stack([xy[:, 0] * self.fx + self.cx,
                                xy[:, 1] * self.fy + self.cy])


def look_at_pose(eye, target, up=(0.0, 0.0, 1.0)) -> SE3:
    """Camera pose (in the world frame) at `eye`, looking at `target`.

    If the viewing direction is parallel to `up` (for example looking straight
    down at the patient) a different reference axis is used automatically.
    """
    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)

    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    if abs(np.dot(z, up)) > 0.999:
        up = np.array([1.0, 0.0, 0.0])

    y = -up - np.dot(-up, z) * z      # "down" direction, orthogonal to z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)

    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return SE3(T, check=False)


class PinholeCamera:
    def __init__(self, intrinsics: CameraIntrinsics, T_mount_cam=None,
                name: str = "camera", near: float = 0.02, far: float = 2.0,
                marker_radius: float = 0.004, noise_std: float = 0.0,
                seed: int | None = 0):
        """
        Parameters
        ----------
        T_mount_cam : pose of the camera in its mounting frame. For the
            eye-in-hand camera this is the (fixed) hand-eye transform in the
            tool frame. For a fixed external camera leave it as the camera
            pose in the world and call world_pose() without arguments.
        marker_radius : physical marker radius [m], used for drawing size.
        noise_std : standard deviation of additive image noise (grey levels).
        """
        self.intrinsics = intrinsics
        self.T_mount_cam = SE3() if T_mount_cam is None else to_se3(T_mount_cam)
        self.name = name
        self.near = near
        self.far = far
        self.marker_radius = marker_radius
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    @property
    def K(self) -> np.ndarray:
        return self.intrinsics.K

    def world_pose(self, T_world_mount=None) -> SE3:
        """Camera pose in the world frame.

        Eye-in-hand: pass robot.fkine(q). External camera: pass nothing.
        """
        base = SE3() if T_world_mount is None else to_se3(T_world_mount)
        return base * self.T_mount_cam

    def project(self, T_world_cam, points_world):
        """Project world points to the image.

        Returns (pixels (N,2), depths (N,), visible (N,) bool). A point is
        visible if it is in front of the camera, between near/far and inside
        the image.
        """
        T_cam_world = to_se3(T_world_cam).inv()
        pixels, depths = project_points(self.K, T_cam_world, points_world)
        w, h = self.intrinsics.width, self.intrinsics.height
        with np.errstate(invalid="ignore"):
            visible = ((depths > self.near) & (depths < self.far)
                    & np.isfinite(pixels).all(axis=1)
                    & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
                    & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
        return pixels, depths, visible

    def depths(self, T_world_cam, points_world) -> np.ndarray:
        """Depth Z of each point in the camera frame (ground truth)."""
        return self.project(T_world_cam, points_world)[1]

    def render(self, T_world_cam, points_world, colors=None) -> np.ndarray:
        """Render a BGR image (H, W, 3, uint8) of the marker points."""
        w, h = self.intrinsics.width, self.intrinsics.height
        img = np.full((h, w, 3), 30, dtype=np.uint8)

        pixels, depths, visible = self.project(T_world_cam, points_world)
        for i in np.argsort(-depths):                    # draw far points first
            if not visible[i]:
                continue
            radius = max(2, int(round(self.intrinsics.fx * self.marker_radius / depths[i])))
            color = colors[i] if colors is not None else marker_bgr(int(i))
            center = (int(round(pixels[i, 0])), int(round(pixels[i, 1])))
            cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)

        if self.noise_std > 0.0:
            noise = self._rng.normal(0.0, self.noise_std, img.shape)
            img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        return img