# src/speed/rules.py
"""
Speed rules + color decision for bounding boxes.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional


def get_max_speed_kmh(label: str, rules: Dict[str, float], default_max: float = 60.0) -> float:
    """
    Returns max speed for a given label from rules table, otherwise default.
    """
    if not label:
        return default_max
    return float(rules.get(label, default_max))


def color_for_speed(
    speed_kmh: Optional[float],
    max_kmh: float,
    close_ratio: float = 0.9
) -> Tuple[int, int, int]:
    """
    OpenCV BGR color:
      - Red if overspeed
      - Yellow if close to max (>= close_ratio * max)
      - Green otherwise
      - Neutral if speed unknown
    """
    if speed_kmh is None or speed_kmh <= 0:
        # unknown speed yet -> draw neutral (blue-ish)
        return (255, 200, 0)

    if speed_kmh > max_kmh:
        return (0, 0, 255)      # Red (BGR)
    if speed_kmh >= close_ratio * max_kmh:
        return (0, 255, 255)    # Yellow
    return (0, 255, 0)          # Green
