"""Coordinate transformation helpers (SE(3)) used across the control package.

Conventions
-----------
* Poses are spatialmath.SE3 objects (4x4 homogeneous matrices).
* Twists/velocities are ordered [vx, vy, vz, wx, wy, wz].
* T_a_b means "pose of frame b expressed in frame a".
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath import SE3


def skew(v) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    x, y, z = np.asarray(v, dtype=float).ravel()
    return np.array([[0.0, -z, y],
                    [z, 0.0, -x],
                    [-y, x, 0.0]])


def to_se3(T) -> SE3:
    """Convert a 4x4 array or SE3 into an SE3 object."""
    if isinstance(T, SE3):
        return T
    return SE3(np.asarray(T, dtype=float), check=False)


def make_pose(xyz=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)) -> SE3:
    """Create an SE3 pose from a position (m) and roll-pitch-yaw (degrees).

    RPY is applied as extrinsic rotations about x, y, z (fixed axes).
    """
    R = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return SE3(T, check=False)


def pose_error(T_current, T_target) -> np.ndarray:
    """6D pose error expressed in the base frame.

    Returns [dx, dy, dz, rx, ry, rz] where the first three entries are the
    position error (target - current) and the last three are the rotation
    error as a rotation vector (axis * angle) taking current to target.
    """
    Tc = to_se3(T_current)
    Tt = to_se3(T_target)
    dp = Tt.t - Tc.t
    R_err = Tt.R @ Tc.R.T
    dr = Rotation.from_matrix(R_err).as_rotvec()
    return np.concatenate([dp, dr])


def adjoint(T) -> np.ndarray:
    """6x6 adjoint matrix of T_a_b.

    Maps a twist expressed in frame b to the same rigid-body twist expressed
    in frame a (including the change of reference point):
        V_a = adjoint(T_a_b) @ V_b
    """
    T = to_se3(T)
    R = T.R
    p = T.t
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def rotate_twist(R: np.ndarray) -> np.ndarray:
    """6x6 block-diagonal matrix that only rotates a twist between frames.

    Use this when the reference point does not change (for example the camera
    velocity expressed in the camera frame -> same point in the base frame).
    """
    R = np.asarray(R, dtype=float)
    M = np.zeros((6, 6))
    M[:3, :3] = R
    M[3:, 3:] = R
    return M


def transform_points(T, points) -> np.ndarray:
    """Apply an SE3 transform to an (N, 3) array of points. Returns (N, 3)."""
    T = to_se3(T)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return pts @ T.R.T + T.t


def project_points(K, T_cam_world, points_world):
    """Project 3D world points onto the image plane (pinhole model).

    Parameters
    ----------
    K : (3, 3) camera intrinsic matrix.
    T_cam_world : pose of the world frame expressed in the camera frame
        (i.e. the inverse of the camera pose in the world).
    points_world : (N, 3) points in the world frame.

    Returns
    -------
    pixels : (N, 2) pixel coordinates (u, v).
    depths : (N,) depth Z of each point in the camera frame.
    Points with non-positive depth are returned as NaN pixels.
    """
    K = np.asarray(K, dtype=float)
    pts_cam = transform_points(T_cam_world, points_world)
    depths = pts_cam[:, 2].copy()

    pixels = np.full((pts_cam.shape[0], 2), np.nan)
    valid = depths > 1e-9
    uvw = pts_cam[valid] @ K.T
    pixels[valid] = uvw[:, :2] / uvw[:, 2:3]
    return pixels, depths