"""Dashboard REST API, recording, and the complete procedure (headless)."""

import json

import numpy as np
import pytest

from control import (
    UR3E_HOME_Q,
    MotionController,
    SafetyConfig,
    SafetyMonitor,
    SafetyState,
    Sphere,
    build_ur3e,
    forward_kinematics,
)
from dashboard import SystemState, create_app, to_jsonable


# ====================================================================
# Dashboard REST API (Flask test client, no sockets)
# ====================================================================
@pytest.fixture
def api():
    state = SystemState(regions=["abdomen", "chest"])
    app = create_app(state)
    app.testing = True
    return state, app.test_client()


def test_to_jsonable_replaces_nan():
    data = to_jsonable({"a": np.array([1.0, np.nan]), "b": float("inf"), "c": np.int64(3)})
    assert data == {"a": [1.0, None], "b": None, "c": 3}
    json.dumps(data, allow_nan=False)


def test_health_and_state(api):
    state, client = api
    assert client.get("/api/health").get_json()["status"] == "ok"
    snapshot = client.get("/api/state").get_json()
    for section in ("status", "patient", "robot", "vision", "safety", "performance", "config"):
        assert section in snapshot


def test_nan_features_become_null(api):
    state, client = api
    state.update_vision(uv=[[1.0, float("nan")]])
    assert client.get("/api/vision").get_json()["current_features"] == [[1.0, None]]


def test_config_validation(api):
    state, client = api
    assert client.put("/api/config", json={"ibvs_gain": 0.8}).status_code == 200
    assert state.get_config()["ibvs_gain"] == 0.8

    assert client.put("/api/config", json={"ibvs_gain": 99}).status_code == 400
    assert client.put("/api/config", json={"bogus": 1}).status_code == 400
    assert client.put("/api/config", json={"ibvs_gain": True}).status_code == 400
    assert state.get_config()["ibvs_gain"] == 0.8                  # unchanged

    response = client.put("/api/config", data="x", content_type="text/plain")
    assert response.status_code == 415 and "error" in response.get_json()


def test_commands(api):
    state, client = api
    assert client.post("/api/command", json={"command": "start"}).status_code == 202
    assert state.pop_commands()[0]["command"] == "start"

    assert client.post("/api/command", json={"command": "explode"}).status_code == 400
    assert client.post("/api/command", json={"command": "select_region"}).status_code == 400


def test_foreign_origins_cannot_send_commands(api):
    state, client = api
    evil = client.post("/api/command", json={"command": "estop"},
                    headers={"Origin": "http://evil.example"})
    assert evil.status_code == 403 and state.pop_commands() == []

    local = client.post("/api/command", json={"command": "pause"},
                        headers={"Origin": "http://localhost:5000"})
    assert local.status_code == 202


def test_cors_headers_only_for_local_origins(api):
    _, client = api
    ok = client.get("/api/health", headers={"Origin": "http://127.0.0.1:5000"})
    assert ok.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1:5000"
    bad = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "Access-Control-Allow-Origin" not in bad.headers


def test_region_endpoint(api):
    state, client = api
    assert client.put("/api/patient/region", json={"region": "chest"}).status_code == 200
    assert state.patient.selected_region == "chest"
    assert state.pop_commands()[-1]["command"] == "select_region"

    assert client.put("/api/patient/region", json={"region": "knee"}).status_code == 400
    state.set_phase("servoing")
    assert client.put("/api/patient/region", json={"region": "abdomen"}).status_code == 409


def test_unknown_api_path_returns_json(api):
    _, client = api
    response = client.get("/api/nope")
    assert response.status_code == 404 and "error" in response.get_json()


# ====================================================================
# Recording
# ====================================================================
def test_recorder_keeps_video_log_and_events_in_sync(tmp_path):
    import cv2
    from recording import CameraCapture, Recorder, SessionManager, sanitize_name

    def camera():
        image = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(image, (80, 60), 20, (0, 255, 0), -1)
        return image

    recorder = Recorder(root=tmp_path, video_fps=25, save_frames_every=5,
                        capture=CameraCapture({"cam": camera}))
    recorder.start(name="t1", metadata={"note": "test"})
    try:
        for k in range(100):
            if k == 40:
                recorder.pause()
            if k == 60:
                recorder.resume()
            recorder.record(k * 0.02, {"a": float(k), "b": 1.0})
    except RuntimeError as exc:                       # no video codec on this machine
        recorder.stop()
        pytest.skip(str(exc))
    recorder.stop()

    manager = SessionManager(tmp_path)
    meta = manager.load_metadata("t1")
    assert meta["state"] == "stopped" and meta["samples"] == 80      # 20 samples were paused

    log = manager.load_robot_log("t1")
    frames = manager.load_frame_index("t1", "cam")
    assert len(log) == 80 and {"a", "b"} <= set(log.columns)
    assert len(frames) == meta["cameras"]["cam"]["frames"] > 0
    assert set(frames["sample"]) <= set(log["sample"])               # every frame has a log row
    merged = frames.merge(log, on="sample", suffixes=("_f", "_l"))
    assert np.allclose(merged["t_sim_f"], merged["t_sim_l"], atol=1e-5)
    assert np.diff(log["t_sim"]).max() > 0.4                         # the pause left a gap

    saved = list((tmp_path / "t1" / "frames" / "cam").glob("*.jpg"))
    assert len(saved) == -(-len(frames) // 5)

    kinds = [e["kind"] for e in manager.load_events("t1")]
    assert kinds == ["start", "pause", "resume", "stop"]

    assert sanitize_name("../../etc") == "etc"
    assert manager.list_sessions()[0]["name"] == "t1"


def test_session_files_cannot_escape_the_session_folder(tmp_path):
    from recording import SessionManager

    manager = SessionManager(tmp_path)
    session = manager.create("safe")
    with pytest.raises(ValueError):
        session.file("..", "outside.txt")
    with pytest.raises(ValueError):
        manager.path_of("../safe")


# ====================================================================
# Controller + safety
# ====================================================================
def _controller(**safety_overrides):
    robot = build_ur3e()
    params = dict(floor_z=None, link_radius=0.0, qdot_max=1.0, tcp_speed_max=1.0)
    params.update(safety_overrides)
    monitor = SafetyMonitor(robot, SafetyConfig(**params))
    steps = []
    ctrl = MotionController(robot, dt=0.02, safety=monitor, on_step=lambda c: steps.append(c.t))
    return robot, monitor, ctrl, steps


def test_joint_move_reaches_the_goal():
    robot, monitor, ctrl, steps = _controller()
    goal = UR3E_HOME_Q + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert ctrl.move_to_joint(goal, duration=1.0)
    assert np.allclose(ctrl.q, goal, atol=5e-3)
    assert len(steps) == 51 and monitor.violation_count == 0


def test_a_zone_on_the_path_stops_the_robot_before_it_enters():
    robot = build_ur3e()
    goal = UR3E_HOME_Q + np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    middle = forward_kinematics(robot, (UR3E_HOME_Q + goal) / 2).t
    _, monitor, ctrl, _ = _controller(restricted_zones=[Sphere("obstacle", middle, 0.03)])

    assert not ctrl.move_to_joint(goal, duration=1.0)
    assert monitor.violation_count >= 1
    assert monitor.check_configuration(ctrl.q).violations == []     # it never entered the zone


# ====================================================================
# Complete procedure (headless, no browser)
# ====================================================================
@pytest.fixture
def app():
    import main

    args = main.parse_args(["--headless", "--no-dashboard", "--max-steps", "1500"])
    application = main.Application(args)
    application.setup()
    yield application
    application.shutdown()


def test_layout_is_reachable(app):
    """If this fails, adjust WorkcellConfig (see the printed table)."""
    assert app.check() == 0


def test_full_procedure_converges(app):
    assert app.run_procedure() == "converged"
    assert app.state.phase == "completed"

    metrics = app.state.performance
    assert metrics["status"] == "converged"
    assert metrics["position_error_mm"] < 10.0
    assert metrics["orientation_error_deg"] < 2.0
    assert metrics["safety_violations"] == 0
    assert metrics["final_ibvs_error_px"] < app.ibvs.tol_px


def test_start_is_refused_while_estop_is_active(app):
    app.monitor.trigger_estop("test")
    assert app.run_procedure() == "estop_active"


def test_stop_command_aborts_the_run(app):
    app.state.push_command("stop")
    assert app.run_procedure() == "stopped"
    assert app.state.phase == "aborted"
    assert not app.running


def test_idle_commands(app):
    app.state.push_command("select_region", {"region": "chest"})
    assert app.poll_commands(idle=True) is False
    assert app.wc.region == "chest"

    app.state.push_command("start")
    assert app.poll_commands(idle=True) is True