# src/speed/speed_estimator.py
"""
Per-track line-crossing speed estimator.

Speed is computed ONLY after:
  - crossing Line 1 segment -> t1 set
  - crossing Line 2 segment -> t2 set
  - speed_kmh = (distance_m / (t2 - t1)) * 3.6

Why some cars were missing speed:
- sign-change detection can miss when the point lands near the line (side ~ 0)
- near-segment margin may be too tight
- tracker jitter -> centroid may skip the line by a few pixels

Fixes:
- crossing condition becomes:
    (sign-change OR "hit" the line: abs(side) <= hit_eps)
  AND the point is near the segment (near_segment(..., margin_px))
- tunable parameters: segment_margin_px, hit_eps, min_dt_sec
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .geometry import Line, side_of_line, crossed, near_segment

Point = Tuple[int, int]


@dataclass
class TrackSpeedState:
    prev_side_l1: Optional[float] = None
    prev_side_l2: Optional[float] = None
    t1: Optional[float] = None
    t2: Optional[float] = None
    speed_kmh: Optional[float] = None
    label: str = ""
    logged: bool = False


class SpeedEstimator:
    def __init__(
        self,
        line1: Line,
        line2: Line,
        distance_m: float,
        segment_margin_px: float = 50.0,
        hit_eps: float = 2.5,
        min_dt_sec: float = 0.15,
    ):
        """
        Args:
            line1, line2: Line segments in image coordinates
            distance_m: real-world distance between lines in meters
            segment_margin_px: how close (pixels) the point must be to the segment to count
            hit_eps: if abs(side_of_line) <= hit_eps, treat as a "hit" (helps when point lands on line)
            min_dt_sec: reject unrealistic extremely small dt (prevents crazy speeds)
        """
        self.line1 = line1
        self.line2 = line2
        self.distance_m = float(distance_m)
        self.segment_margin_px = float(segment_margin_px)
        self.hit_eps = float(hit_eps)
        self.min_dt_sec = float(min_dt_sec)

        self.states: Dict[int, TrackSpeedState] = {}

    @staticmethod
    def bbox_centroid(x1: int, y1: int, x2: int, y2: int) -> Point:
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _cross_or_hit(self, prev_side: Optional[float], curr_side: float) -> bool:
        """
        Returns True if:
          - sign changed (cross), OR
          - point is very close to the line (hit)
        """
        if abs(curr_side) <= self.hit_eps:
            return True
        return crossed(prev_side, curr_side)

    def update(self, track_id: int, centroid: Point, t_now: float, label: str) -> TrackSpeedState:
        st = self.states.get(track_id)
        if st is None:
            st = TrackSpeedState(label=label)
            self.states[track_id] = st

        if label:
            st.label = label

        curr_l1 = side_of_line(self.line1, centroid)
        curr_l2 = side_of_line(self.line2, centroid)

        # ---- Line 1: set t1 once ----
        if st.t1 is None:
            if (
                self._cross_or_hit(st.prev_side_l1, curr_l1)
                and near_segment(self.line1, centroid, margin_px=self.segment_margin_px)
            ):
                st.t1 = t_now

        # ---- Line 2: set t2 once, compute speed once ----
        if st.t1 is not None and st.speed_kmh is None:
            if (
                self._cross_or_hit(st.prev_side_l2, curr_l2)
                and near_segment(self.line2, centroid, margin_px=self.segment_margin_px)
            ):
                st.t2 = t_now
                dt = st.t2 - st.t1

                # Guard against tiny dt producing insane speeds
                if dt >= self.min_dt_sec:
                    st.speed_kmh = (self.distance_m / dt) * 3.6

        st.prev_side_l1 = curr_l1
        st.prev_side_l2 = curr_l2

        return st
