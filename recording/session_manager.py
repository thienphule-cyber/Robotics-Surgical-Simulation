"""Recording sessions: folders, metadata and loading data back for analysis.

Layout of one session folder:
    metadata.json, robot_log.csv, events.jsonl,
    <camera>.mp4 + <camera>_frames.csv (one pair per camera),
    frames/<camera>/ (optional images), snapshots/
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

METADATA_FILE = "metadata.json"
ROBOT_LOG_FILE = "robot_log.csv"
EVENTS_FILE = "events.jsonl"


def frame_index_file(camera: str) -> str:
    return f"{camera}_frames.csv"


def sanitize_name(name: str) -> str:
    """Make a name safe to use as a file or folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return cleaned or "unnamed"


def json_default(obj):
    """json.dumps helper for numpy values, paths and enums."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


@dataclass
class Session:
    name: str
    path: Path
    created_at: float
    metadata: dict = field(default_factory=dict)

    def file(self, *parts: str) -> Path:
        """Path of a file inside the session (parent folders are created)."""
        target = self.path.joinpath(*parts).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise ValueError("Path is outside the session folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def dir(self, *parts: str) -> Path:
        """Path of a folder inside the session (created if missing)."""
        target = self.path.joinpath(*parts).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise ValueError("Path is outside the session folder")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_metadata(self):
        """Write metadata.json atomically."""
        target = self.path / METADATA_FILE
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.metadata, indent=2, default=json_default))
        os.replace(tmp, target)

    def update_metadata(self, **kwargs):
        self.metadata.update(kwargs)
        self.write_metadata()


class SessionManager:
    def __init__(self, root="recordings"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, name: str | None = None, metadata: dict | None = None) -> Session:
        """Create a new session folder (a numeric suffix avoids collisions)."""
        base = sanitize_name(name) if name else datetime.now().strftime("session_%Y%m%d_%H%M%S")
        candidate, n = base, 2
        while (self.root / candidate).exists():
            candidate = f"{base}_{n}"
            n += 1
        path = self.root / candidate
        path.mkdir(parents=True)

        created = time.time()
        meta = {"name": candidate, "created_at": _iso(created), "state": "created"}
        meta.update(metadata or {})
        session = Session(candidate, path, created, meta)
        session.write_metadata()
        return session

    # ------------------------------------------------------------------
    # Reading sessions back
    # ------------------------------------------------------------------
    def path_of(self, name: str) -> Path:
        if sanitize_name(name) != name:
            raise ValueError(f"Invalid session name '{name}'")
        path = self.root / name
        if not path.is_dir():
            raise FileNotFoundError(f"No such session: {name}")
        return path

    def list_sessions(self) -> list:
        """Metadata of every session, oldest first."""
        sessions = []
        for path in sorted(p for p in self.root.iterdir() if p.is_dir()):
            meta_file = path / METADATA_FILE
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text())
            except json.JSONDecodeError:
                continue
            meta.setdefault("name", path.name)
            meta["path"] = str(path)
            sessions.append(meta)
        return sessions

    def latest(self) -> str | None:
        sessions = self.list_sessions()
        return sessions[-1]["name"] if sessions else None

    def load_metadata(self, name: str) -> dict:
        return json.loads((self.path_of(name) / METADATA_FILE).read_text())

    def load_robot_log(self, name: str):
        """Robot/control log as a pandas DataFrame."""
        import pandas as pd
        return pd.read_csv(self.path_of(name) / ROBOT_LOG_FILE)

    def load_frame_index(self, name: str, camera: str):
        """Per-frame timestamps (frame, sample, t_sim, t_wall, image_file)."""
        import pandas as pd
        return pd.read_csv(self.path_of(name) / frame_index_file(sanitize_name(camera)))

    def load_events(self, name: str) -> list:
        path = self.path_of(name) / EVENTS_FILE
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]