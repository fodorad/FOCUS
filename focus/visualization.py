"""Frame overlay drawing for the processed camera feed."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

    from focus.engine import EngineResult


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_BBOX = (0, 255, 0)  # green
COLOR_LANDMARK = (255, 200, 0)  # cyan-ish
COLOR_GAZE = (0, 255, 255)  # yellow
COLOR_AXIS_X = (0, 0, 255)  # red   (yaw)
COLOR_AXIS_Y = (0, 255, 0)  # green (pitch)
COLOR_AXIS_Z = (255, 0, 0)  # blue  (roll)
COLOR_TEXT_BG = (30, 30, 30)
COLOR_TEXT_FG = (220, 220, 220)


def draw_overlays(frame: np.ndarray, result: EngineResult) -> np.ndarray:
    """Draw all visual overlays on a copy of the frame.

    Args:
        frame: BGR image, shape (H, W, 3), dtype uint8.
        result: EngineResult from the processing engine.

    Returns:
        Annotated BGR image (same shape).
    """
    out = frame.copy()
    _draw_bbox(out, result)
    _draw_landmarks(out, result)
    _draw_gaze(out, result)
    _draw_head_pose_axes(out, result)
    _draw_info_overlay(out, result)
    return out


def _draw_bbox(frame: np.ndarray, result: EngineResult) -> None:
    x1, y1, x2, y2 = result.face.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)
    label = f"face {result.face.confidence:.0%}"
    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BBOX, 1)


def _draw_landmarks(frame: np.ndarray, result: EngineResult) -> None:
    for x, y in result.landmarks.points:
        cv2.circle(frame, (int(x), int(y)), 2, COLOR_LANDMARK, -1)


def _draw_gaze(frame: np.ndarray, result: EngineResult) -> None:
    """Draw gaze direction arrows from eye region centres."""
    x1, y1, x2, y2 = result.face.bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = x2 - x1

    length = bw * 0.6
    pitch_rad = math.radians(result.gaze.pitch)
    yaw_rad = math.radians(result.gaze.yaw)

    dx = int(length * math.sin(yaw_rad))
    dy = int(-length * math.sin(pitch_rad))

    # left eye estimate
    lx, ly = cx - bw // 5, cy - (y2 - y1) // 10
    cv2.arrowedLine(frame, (lx, ly), (lx + dx, ly + dy), COLOR_GAZE, 2, tipLength=0.25)

    # right eye estimate
    rx, ry = cx + bw // 5, cy - (y2 - y1) // 10
    cv2.arrowedLine(frame, (rx, ry), (rx + dx, ry + dy), COLOR_GAZE, 2, tipLength=0.25)


def _draw_head_pose_axes(frame: np.ndarray, result: EngineResult) -> None:
    """Draw RGB axes indicating head orientation."""
    x1, y1, x2, y2 = result.face.bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    axis_len = (x2 - x1) * 0.5

    yaw = math.radians(result.head_pose.yaw)
    pitch = math.radians(result.head_pose.pitch)
    roll = math.radians(result.head_pose.roll)

    # X axis (yaw) — red
    ex = int(cx + axis_len * math.cos(yaw) * math.cos(roll))
    ey = int(cy + axis_len * math.sin(roll))
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), COLOR_AXIS_X, 2, tipLength=0.2)

    # Y axis (pitch) — green
    ex2 = int(cx - axis_len * math.sin(roll))
    ey2 = int(cy - axis_len * math.cos(pitch) * math.cos(roll))
    cv2.arrowedLine(frame, (cx, cy), (ex2, ey2), COLOR_AXIS_Y, 2, tipLength=0.2)

    # Z axis (roll) — blue (depth, foreshortened)
    ez_len = axis_len * 0.5
    ex3 = int(cx + ez_len * math.sin(yaw))
    ey3 = int(cy - ez_len * math.sin(pitch))
    cv2.arrowedLine(frame, (cx, cy), (ex3, ey3), COLOR_AXIS_Z, 2, tipLength=0.2)


def _draw_info_overlay(frame: np.ndarray, result: EngineResult) -> None:
    """Small text overlay in top-left corner."""
    lines = [
        f"Engagement: {result.engagement_score:.0f}",
        f"Eye contact: {result.eye_contact_pct:.0f}%",
        f"Gaze: P={result.gaze.pitch:+.1f} Y={result.gaze.yaw:+.1f}",
        f"Head: Y={result.head_pose.yaw:+.1f} P={result.head_pose.pitch:+.1f} "
        f"R={result.head_pose.roll:+.1f}",
    ]
    y0 = 20
    for i, line in enumerate(lines):
        y = y0 + i * 20
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_BG, 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_FG, 1)
