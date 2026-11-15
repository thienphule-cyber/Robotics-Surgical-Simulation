"""Flask app + WebSocket server running in background threads.

The Swift simulation keeps the main thread. DashboardServer starts:
  * the Flask REST API (and the static frontend) on http_port
  * the WebSocket hub on ws_port
Both bind to 127.0.0.1 by default (local use only).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server

from ..data.patient_state import SystemState
from .routes import create_blueprint
from .websocket import WebSocketHub

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Only pages served from localhost may call the API from a browser.
_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")


def create_app(state: SystemState, ws_port: int = 8765) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["WS_PORT"] = ws_port
    app.register_blueprint(create_blueprint(state))

    @app.before_request
    def _check_origin():
        # Reject state-changing requests coming from other websites.
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = request.headers.get("Origin")
            if origin and not _ORIGIN_RE.match(origin):
                abort(403, description="Origin not allowed")

    @app.after_request
    def _add_headers(response):
        origin = request.headers.get("Origin")
        if origin and _ORIGIN_RE.match(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        if request.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(HTTPException)
    def _json_errors(exc):
        if request.path.startswith("/api"):
            return jsonify({"error": exc.description, "status": exc.code}), exc.code
        return exc

    @app.get("/")
    def index():
        if not (FRONTEND_DIR / "index.html").exists():
            return jsonify({"message": "Frontend not found. Create dashboard/frontend/index.html.",
                            "api": "/api/state"}), 404
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/<path:filename>")
    def frontend_file(filename):
        return send_from_directory(FRONTEND_DIR, filename)

    return app


class DashboardServer:
    def __init__(self, state: SystemState | None = None,
                 host: str = "127.0.0.1", http_port: int = 5000,
                 ws_port: int = 8765, ws_rate_hz: float = 10.0):
        self.state = state or SystemState()
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port

        origins = [None, f"http://localhost:{http_port}", f"http://127.0.0.1:{http_port}"]
        self.hub = WebSocketHub(self.state, rate_hz=ws_rate_hz, allowed_origins=origins)

        self._http = None
        self._http_thread = None
        self._ws_thread = None
        self._loop = None
        self._ws_stop = None
        self._ws_ready = threading.Event()
        self._ws_error = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    # ------------------------------------------------------------------
    def start(self):
        if self._http is not None:
            return
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        app = create_app(self.state, self.ws_port)
        self._http = make_server(self.host, self.http_port, app, threaded=True)

        self._ws_ready.clear()
        self._ws_error = None
        self._ws_thread = threading.Thread(target=self._run_ws, name="dashboard-ws", daemon=True)
        self._ws_thread.start()
        if not self._ws_ready.wait(timeout=5.0) or self._ws_error is not None:
            self._http.server_close()
            self._http = None
            raise RuntimeError(f"WebSocket server failed to start: {self._ws_error}")

        self._http_thread = threading.Thread(target=self._http.serve_forever,
                                             name="dashboard-http", daemon=True)
        self._http_thread.start()
        self.state.log(f"Dashboard running at {self.url} (WebSocket port {self.ws_port})",
                       source="dashboard")

    def _run_ws(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._ws_main())
        except Exception as exc:          # for example the port is already in use
            self._ws_error = exc
            self._ws_ready.set()
        finally:
            loop.close()

    async def _ws_main(self):
        self._ws_stop = asyncio.Event()
        # Signal readiness shortly after the server starts listening.
        asyncio.get_running_loop().call_later(0.2, self._ws_ready.set)
        await self.hub.run(self.host, self.ws_port, self._ws_stop)

    def stop(self):
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
            self._http = None
        if self._loop is not None and self._ws_stop is not None:
            self._loop.call_soon_threadsafe(self._ws_stop.set)
        for thread in (self._http_thread, self._ws_thread):
            if thread is not None:
                thread.join(timeout=3.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()