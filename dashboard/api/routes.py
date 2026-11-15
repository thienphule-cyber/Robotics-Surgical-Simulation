"""REST API routes (Flask blueprint).

Read-only data:  GET /api/state, /api/status, /api/patient, /api/robot,
                 /api/vision, /api/safety, /api/performance, /api/logs
Parameters:      GET/PUT /api/config
Patient:         PUT /api/patient/region
Commands:        POST /api/command   (queued, executed by the main loop)
"""

from __future__ import annotations

import time

from flask import Blueprint, abort, current_app, jsonify, request

from ..data.patient_state import ACTIVE_PHASES, CONFIG_SPEC, LOG_LEVELS

SECTIONS = ("status", "patient", "robot", "vision", "safety", "performance")

# Commands the dashboard is allowed to send (select_region has its own route).
USER_COMMANDS = ("start", "pause", "resume", "stop", "estop", "reset_estop",
                 "reset", "start_recording", "stop_recording")


def _json_body() -> dict:
    """Parsed JSON object. Flask answers 415 if the content type is not JSON."""
    body = request.get_json()
    if not isinstance(body, dict):
        abort(400, description="JSON body must be an object")
    return body


def create_blueprint(state) -> Blueprint:
    bp = Blueprint("api", __name__, url_prefix="/api")

    @bp.get("/health")
    def health():
        return jsonify({"status": "ok", "version": state.version, "time": time.time()})

    @bp.get("/info")
    def info():
        return jsonify({
            "name": "UR3e Vision-Guided Robotic Surgical Simulation",
            "ws_port": current_app.config.get("WS_PORT"),
            "commands": list(USER_COMMANDS),
            "config_spec": {k: {"min": v[0], "max": v[1], "default": v[2]}
                            for k, v in CONFIG_SPEC.items()},
        })

    @bp.get("/state")
    def full_state():
        return jsonify(state.snapshot())

    def _make_section_view(name):
        def view():
            return jsonify(state.section(name))
        view.__name__ = f"get_{name}"
        return view

    for _name in SECTIONS:
        bp.add_url_rule(f"/{_name}", view_func=_make_section_view(_name),
                        methods=["GET"])

    @bp.get("/logs")
    def get_logs():
        limit = request.args.get("limit", default=100, type=int)
        limit = max(1, min(limit, 500))
        level = request.args.get("level")
        if level is not None and level not in LOG_LEVELS:
            abort(400, description=f"level must be one of {list(LOG_LEVELS)}")
        return jsonify({"entries": state.recent_logs(limit, level)})

    @bp.get("/config")
    def get_config():
        return jsonify({"config": state.get_config(),
                        "version": state.config_version})

    @bp.put("/config")
    def put_config():
        body = _json_body()
        try:
            state.set_config(body)
        except ValueError as exc:
            abort(400, description=str(exc))
        return jsonify({"config": state.get_config(),
                        "version": state.config_version})

    @bp.put("/patient/region")
    def put_region():
        body = _json_body()
        region = body.get("region")
        if not isinstance(region, str) or region not in state.patient.available_regions:
            abort(400, description="Unknown region. Available: "
                  f"{state.patient.available_regions}")
        if state.phase in ACTIVE_PHASES:
            abort(409, description="Cannot change the region while the robot is moving")
        state.set_region(region)
        state.push_command("select_region", {"region": region})
        return jsonify(state.section("patient"))

    @bp.post("/command")
    def post_command():
        body = _json_body()
        command = body.get("command")
        if command not in USER_COMMANDS:
            abort(400, description=f"command must be one of {list(USER_COMMANDS)}")
        payload = body.get("payload", {})
        if not isinstance(payload, dict):
            abort(400, description="payload must be an object")
        state.push_command(command, payload)
        return jsonify({"queued": command}), 202

    return bp