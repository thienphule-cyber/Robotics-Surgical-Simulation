"""WebSocket hub: streams the system state to every connected dashboard.

Messages are JSON objects {"type", "seq", "timestamp", "data"}:
  "snapshot": full state, sent once when a client connects
  "state":    full state without logs, sent at `rate_hz`
  "log":      new log entries ({"entries": [...]}); a client that has just
              connected may receive an entry twice, so dedupe by "id"
  "pong":     answer to the text message "ping"
The stream is read-only: commands go through the REST API.
"""

from __future__ import annotations

import asyncio
import json
import time

from websockets.asyncio.server import broadcast, serve
from websockets.exceptions import ConnectionClosed

from ..data.patient_state import to_jsonable


class WebSocketHub:
    def __init__(self, state, rate_hz: float = 10.0, allowed_origins=None):
        """
        allowed_origins : list of accepted browser origins (use None inside
            the list to also accept clients that send no Origin header).
            None disables the check.
        """
        self.state = state
        self.rate_hz = rate_hz
        self.allowed_origins = allowed_origins
        self.clients = set()
        self._seq = 0

    def _encode(self, msg_type: str, data) -> str:
        self._seq += 1
        return json.dumps({"type": msg_type, "seq": self._seq,
                           "timestamp": time.time(), "data": to_jsonable(data)})

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.send(self._encode("snapshot", self.state.snapshot()))
            async for message in websocket:
                if message == "ping":
                    await websocket.send(self._encode("pong", {}))
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)

    async def _broadcaster(self):
        period = 1.0 / self.rate_hz
        last_log_id = self.state.last_log_id
        while True:
            await asyncio.sleep(period)
            self.state.tick_vitals()

            if not self.clients:
                last_log_id = self.state.last_log_id
                continue

            broadcast(self.clients,
                    self._encode("state", self.state.snapshot(include_logs=False)))

            entries = self.state.logs_since(last_log_id)
            if entries:
                broadcast(self.clients, self._encode("log", {"entries": entries}))
                last_log_id = entries[-1]["id"]

    async def run(self, host: str, port: int, stop_event: asyncio.Event):
        """Serve until stop_event is set."""
        async with serve(self.handler, host, port,
                         origins=self.allowed_origins, max_size=2 ** 16):
            task = asyncio.create_task(self._broadcaster())
            await stop_event.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass