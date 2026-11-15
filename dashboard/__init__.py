"""Dashboard package: shared state, REST API and WebSocket streaming."""

from .data.patient_state import (
    Phase,
    PatientState,
    SystemState,
    to_jsonable,
)
from .api import (
    DashboardServer,
    WebSocketHub,
    create_app,
    create_blueprint,
)


def start_dashboard(state: SystemState | None = None, **kwargs) -> DashboardServer:
    """Create a DashboardServer, start it and return it.

    kwargs are passed to DashboardServer (host, http_port, ws_port, ws_rate_hz).
    The shared state is available as `server.state`.
    """
    server = DashboardServer(state=state, **kwargs)
    server.start()
    return server


__all__ = [
    "Phase", "PatientState", "SystemState", "to_jsonable",
    "DashboardServer", "WebSocketHub", "create_app", "create_blueprint",
    "start_dashboard",
]