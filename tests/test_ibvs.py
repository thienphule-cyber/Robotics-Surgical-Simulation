"""IBVS control law, camera model, feature detection and tracking."""

import numpy as np
import pytest
from spatialmath import SE3

from control import (
    BlobDetector,
    CameraIntrinsics,
    ColorMarkerDetector,
    Detection,
    FeatureTracker,
    IBVSController,
    PinholeCamera,
    desired_from_pose,
    interaction_matrix,
    look_at_pose,
    pose_error,
    transform_points,
)

INTR = CameraIntrinsics.from_fov(640, 480, 80.0)
POINTS = np.array([[-0.03, -0.03, 0.30], [0.03, -0.03, 0.30],
                [0.03, 0.03, 0.30], [-0.03, 0.03, 0.30]])   # square, 30 cm in front of T_DES
T_DES = SE3()


@pytest.fixture
def cam():
    return PinholeCamera(INTR)


@pytest.fixture
def desired(cam):
    return desired_from_pose(cam, T_DES, POINTS)


# ---------------------------------------------------------------- control law
def test_interaction_matrix_matches_finite_differences():
    P = np.array([0.05, -0.03, 0.4])                       # point in the camera frame
    twist = np.array([0.02, -0.01, 0.03, 0.05, -0.04, 0.02])
    dt = 1e-6
    p_dot = -twist[:3] - np.cross(twist[3:], P)            # motion of a fixed point seen by a moving camera
    P1 = P + p_dot * dt
    numeric = (P1[:2] / P1[2] - P[:2] / P[2]) / dt
    L = interaction_matrix((P[:2] / P[2])[None, :], P[2])
    assert np.allclose(L @ twist, numeric, atol=1e-5)


def _servo(ibvs, cam, T_start, steps=600, dt=0.05):
    T = T_start
    res = None
    for k in range(steps):
        uv, depth, visible = cam.project(T, POINTS)
        assert visible.all(), "a marker left the field of view"
        res = ibvs.compute(uv, depth)
        if res.converged:
            return T, res
        T = T * SE3.Exp(res.twist * dt)
    return T, res


@pytest.mark.parametrize("mode", ["current", "desired", "mean"])
def test_ibvs_converges_to_the_desired_pose(cam, desired, mode):
    ibvs = IBVSController(INTR, gain=0.8, depth_mode=mode, tol_px=0.5)
    ibvs.set_desired(*desired)
    T_start = T_DES * SE3.Trans(0.04, -0.03, -0.06) * SE3.Rz(0.15)

    T_final, res = _servo(ibvs, cam, T_start)

    assert res.converged
    err = pose_error(T_final, T_DES)
    assert np.linalg.norm(err[:3]) < 1e-3
    assert np.linalg.norm(err[3:]) < 0.01


def test_no_motion_at_the_desired_view(cam, desired):
    ibvs = IBVSController(INTR)
    ibvs.set_desired(*desired)
    uv, depth, _ = cam.project(T_DES, POINTS)
    res = ibvs.compute(uv, depth)
    assert res.converged and np.allclose(res.twist, 0.0)


def test_velocity_limits(cam, desired):
    ibvs = IBVSController(INTR, gain=5.0, v_max=0.02, w_max=0.1)
    ibvs.set_desired(*desired)
    uv, depth, _ = cam.project(T_DES * SE3.Trans(0.05, 0.03, -0.1) * SE3.Rz(0.4), POINTS)
    res = ibvs.compute(uv, depth)
    assert np.linalg.norm(res.twist[:3]) <= 0.02 + 1e-9
    assert np.linalg.norm(res.twist[3:]) <= 0.1 + 1e-9
    assert np.linalg.norm(res.twist) > 0


def test_lost_features(cam, desired):
    ibvs = IBVSController(INTR)
    ibvs.set_desired(*desired)
    uv, depth, _ = cam.project(T_DES * SE3.Trans(0.02, 0.0, -0.03), POINTS)

    uv3 = uv.copy()
    uv3[1] = np.nan                                         # 3 features left: still works
    res = ibvs.compute(uv3, depth)
    assert res.status == "running" and np.all(np.isfinite(res.twist))

    uv2 = uv3.copy()
    uv2[2] = np.nan                                         # 2 features left: not enough
    res = ibvs.compute(uv2, depth)
    assert res.status == "insufficient_features" and np.allclose(res.twist, 0.0)


def test_compute_requires_a_desired_view():
    with pytest.raises(RuntimeError):
        IBVSController(INTR).compute(np.zeros((4, 2)))


# ---------------------------------------------------------------- camera
def test_look_at_pose_points_at_the_target():
    for eye in ([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.3, -0.2, 0.5]):
        T = look_at_pose(eye, [0.0, 0.0, 0.0])
        assert np.allclose(T.R.T @ T.R, np.eye(3), atol=1e-9)
        assert np.linalg.det(T.R) == pytest.approx(1.0)
        target_in_cam = transform_points(T.inv(), np.zeros((1, 3)))[0]
        assert np.allclose(target_in_cam[:2], 0.0, atol=1e-9)
        assert target_in_cam[2] == pytest.approx(np.linalg.norm(eye))


def test_visibility_mask(cam):
    pts = np.array([[0.0, 0.0, 0.3],       # visible
                    [0.0, 0.0, -0.3],      # behind the camera
                    [5.0, 0.0, 0.3],       # outside the image
                    [0.0, 0.0, 5.0]])      # beyond the far plane
    _, _, visible = cam.project(SE3(), pts)
    assert visible.tolist() == [True, False, False, False]


# ---------------------------------------------------------------- detection and tracking
def test_color_detector_finds_all_markers(cam):
    image = cam.render(SE3(), POINTS)
    detections = ColorMarkerDetector(n_markers=4).detect(image)
    assert [d.id for d in detections] == [0, 1, 2, 3]
    pixels, _, _ = cam.project(SE3(), POINTS)
    for d in detections:
        assert np.linalg.norm(d.uv - pixels[d.id]) < 1.5


def test_color_detector_with_noise():
    noisy = PinholeCamera(INTR, noise_std=3.0)
    detections = ColorMarkerDetector(n_markers=4).detect(noisy.render(SE3(), POINTS))
    assert len(detections) == 4


def test_color_detector_misses_markers_outside_the_view(cam):
    far = POINTS + np.array([10.0, 0.0, 0.0])
    assert ColorMarkerDetector(n_markers=4).detect(cam.render(SE3(), far)) == []
    assert ColorMarkerDetector(n_markers=4).detect(np.full((120, 160, 3), 30, np.uint8)) == []


def test_blob_detector_with_reference_initialisation(cam):
    image = cam.render(SE3(), POINTS)
    detections = BlobDetector().detect(image)
    assert len(detections) == 4

    pixels, _, _ = cam.project(SE3(), POINTS)
    tracker = FeatureTracker()
    tracker.initialize_from_reference(detections, pixels + 3.0)    # reference slightly off
    for i in range(4):
        assert np.linalg.norm(tracker.tracks[i].uv - pixels[i]) < 1.5


def test_tracker_coasts_then_drops_lost_features():
    tracker = FeatureTracker(max_lost=3, alpha=1.0, beta=0.0)
    tracker.update([Detection(0, np.array([10.0, 20.0]), 5.0)])
    assert np.allclose(tracker.positions([0]), [[10.0, 20.0]])

    for _ in range(3):
        tracker.update([])
    assert np.all(np.isfinite(tracker.positions([0])))            # still coasting

    tracker.update([])
    assert np.all(np.isnan(tracker.positions([0])))               # dropped


def test_tracker_without_smoothing_returns_the_measurement():
    tracker = FeatureTracker(alpha=1.0, beta=0.0)
    tracker.update([Detection(1, np.array([5.0, 5.0]), 4.0)])
    tracker.update([Detection(1, np.array([8.0, 6.0]), 4.0)])
    assert np.allclose(tracker.positions([1]), [[8.0, 6.0]])


def test_tracker_keeps_identities_for_unlabeled_detections():
    tracker = FeatureTracker(alpha=1.0, beta=0.0)
    tracker.update([Detection(None, np.array([10.0, 10.0]), 4.0),
                    Detection(None, np.array([100.0, 100.0]), 4.0)])
    # next frame: both moved by 2 px and the detection order is swapped
    tracker.update([Detection(None, np.array([102.0, 102.0]), 4.0),
                    Detection(None, np.array([12.0, 12.0]), 4.0)])
    assert np.allclose(tracker.tracks[0].uv, [12.0, 12.0])
    assert np.allclose(tracker.tracks[1].uv, [102.0, 102.0])


def test_tracker_gate_rejects_far_detections():
    tracker = FeatureTracker(max_jump=20.0, alpha=1.0, beta=0.0)
    tracker.update([Detection(None, np.array([0.0, 0.0]), 4.0)])
    tracker.update([Detection(None, np.array([200.0, 200.0]), 4.0)])
    assert len(tracker.tracks) == 2                                # new track, old one coasts
    assert tracker.tracks[0].lost == 1