"""Feature tracking: keeps feature identities and positions between frames.

* Alpha-beta filter (constant velocity) smooths and predicts positions.
* Labeled detections (id given) update their track directly.
* Unlabeled detections are matched to tracks with the Hungarian algorithm,
    with a maximum jump (gate) in pixels.
* A feature that disappears is coasted for `max_lost` frames, then dropped
    (its position is returned as NaN).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    uv: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lost: int = 0        # consecutive frames without a measurement
    hits: int = 1        # total number of measurements received


class FeatureTracker:
    def __init__(self, max_jump: float = 60.0, max_lost: int = 10,
                alpha: float = 0.85, beta: float = 0.1, spawn_new: bool = True):
        """
        alpha, beta : alpha-beta filter gains. Use alpha=1.0, beta=0.0 to
            disable smoothing (measurement is used as-is, no lag).
        max_jump : largest allowed distance (px) between prediction and an
            unlabeled detection to accept the match.
        """
        self.max_jump = max_jump
        self.max_lost = max_lost
        self.alpha = alpha
        self.beta = beta
        self.spawn_new = spawn_new
        self.tracks = {}
        self._next_id = 0

    def reset(self):
        self.tracks = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    def _predict(self, tr: Track) -> np.ndarray:
        return tr.uv + tr.vel

    def _correct(self, tr: Track, meas: np.ndarray):
        pred = self._predict(tr)
        r = meas - pred
        tr.uv = pred + self.alpha * r
        tr.vel = tr.vel + self.beta * r
        tr.lost = 0
        tr.hits += 1

    def _miss(self, tr: Track):
        tr.uv = self._predict(tr)     # coast on the predicted position
        tr.vel = tr.vel * 0.9
        tr.lost += 1

    def _new_track(self, uv: np.ndarray, track_id: int | None = None) -> int:
        tid = self._next_id if track_id is None else track_id
        self.tracks[tid] = Track(tid, np.asarray(uv, dtype=float).copy())
        self._next_id = max(self._next_id, tid + 1)
        return tid

    # ------------------------------------------------------------------
    def update(self, detections: list):
        """Process the detections of one frame."""
        updated = set()
        labeled = [d for d in detections if d.id is not None]
        unlabeled = [d for d in detections if d.id is None]

        # 1) Labeled detections update their own track.
        for d in labeled:
            if d.id in self.tracks:
                self._correct(self.tracks[d.id], d.uv)
            else:
                self._new_track(d.uv, d.id)
            updated.add(d.id)

        # 2) Associate unlabeled detections with the remaining tracks.
        matched = set()
        free_ids = [i for i in self.tracks if i not in updated]
        if free_ids and unlabeled:
            preds = np.array([self._predict(self.tracks[i]) for i in free_ids])
            dets = np.array([d.uv for d in unlabeled])
            cost = np.linalg.norm(preds[:, None, :] - dets[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.max_jump:
                    self._correct(self.tracks[free_ids[r]], dets[c])
                    updated.add(free_ids[r])
                    matched.add(c)

        # 3) Unmatched unlabeled detections start new tracks.
        if self.spawn_new:
            for j, d in enumerate(unlabeled):
                if j not in matched:
                    updated.add(self._new_track(d.uv))

        # 4) Tracks without a measurement coast.
        for tid, tr in self.tracks.items():
            if tid not in updated:
                self._miss(tr)

    def initialize_from_reference(self, detections: list, reference_uv):
        """Give unlabeled detections ids by matching them to reference points.

        Track i is assigned to the detection closest to reference_uv[i]
        (Hungarian matching). Only reliable if the initial view is not far
        from the reference view; prefer ColorMarkerDetector otherwise.
        """
        self.reset()
        ref = np.asarray(reference_uv, dtype=float)
        if not detections:
            return
        dets = np.array([d.uv for d in detections])
        cost = np.linalg.norm(ref[:, None, :] - dets[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            self._new_track(dets[c], int(r))
        self._next_id = max(self._next_id, len(ref))

    # ------------------------------------------------------------------
    def positions(self, ids) -> np.ndarray:
        """(len(ids), 2) positions; NaN rows for features that are lost."""
        ids = list(ids)
        out = np.full((len(ids), 2), np.nan)
        for k, i in enumerate(ids):
            tr = self.tracks.get(i)
            if tr is not None and tr.lost <= self.max_lost:
                out[k] = tr.uv
        return out

    def active_ids(self) -> list:
        return [i for i, tr in self.tracks.items() if tr.lost <= self.max_lost]