# src/speed/geometry.py
"""
Geometry helpers for line-crossing speed estimation.

We detect line crossing by checking the sign change of the point-side-of-line
value between consecutive frames.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

Point = Tuple[int, int]


@dataclass(frozen=True)
class Line:
    """A 2D line represented by two points (A -> B) in image coordinates."""
    a: Point
    b: Point


def side_of_line(line: Line, p: Point) -> float:
    """
    Returns the signed area (cross product) that indicates which side of the line point p lies on.
    Positive/negative indicates side; 0 means on the line (numerically unlikely).
    """
    ax, ay = line.a
    bx, by = line.b
    px, py = p
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def crossed(prev_side: Optional[float], curr_side: float, eps: float = 1e-6) -> bool:
    """
    Returns True if the sign changed between prev_side and curr_side (line crossing).
    eps avoids jitter when values are near 0.
    """
    if prev_side is None:
        return False
    if abs(prev_side) < eps or abs(curr_side) < eps:
        return False
    return (prev_side > 0 and curr_side < 0) or (prev_side < 0 and curr_side > 0)

def near_segment(line: Line, p: Point, margin_px: float = 25.0) -> bool:
    """
    Returns True if point p is close to the LINE SEGMENT (not infinite line).
    margin_px controls tolerance.
    """
    ax, ay = line.a
    bx, by = line.b
    px, py = p

    # segment vector
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay

    # segment length^2
    vv = vx * vx + vy * vy
    if vv < 1e-6:
        return False

    # projection factor t in [0,1] for closest point on segment
    t = (wx * vx + wy * vy) / vv
    if t < 0.0:
        closest = (ax, ay)
    elif t > 1.0:
        closest = (bx, by)
    else:
        closest = (ax + t * vx, ay + t * vy)

    cx, cy = closest
    dx = px - cx
    dy = py - cy
    dist = (dx * dx + dy * dy) ** 0.5

    return dist <= margin_px

