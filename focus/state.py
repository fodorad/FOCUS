"""Session state management with rolling metric buffers for real-time tracking."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

ENGAGEMENT_THRESHOLDS: list[tuple[int, str, str]] = [
    (20, "Very Low", "#e74c3c"),  # red
    (40, "Low", "#e67e22"),  # orange
    (60, "Neutral", "#3498db"),  # blue
    (80, "High", "#27ae60"),  # dark green
    (100, "Very High", "#2ecc71"),  # light green
]

WINDOW_SECONDS = 30


def engagement_label(score: float) -> tuple[str, str]:
    """Map engagement score (0-100) to (label, hex_color)."""
    for threshold, label, color in ENGAGEMENT_THRESHOLDS:
        if score <= threshold:
            return label, color
    return "Very High", "#2ecc71"


@dataclass
class MetricSnapshot:
    """Single point-in-time metric reading."""

    timestamp: float
    engagement_score: float = 0.0
    eye_contact_pct: float = 0.0
    blink_event: bool = False
    blink_rate: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    speaking_ratio: float = 0.0
    dominant_au: str = "N/A"


class SessionState:
    """Thread-safe rolling buffer of metrics for the last WINDOW_SECONDS seconds."""

    def __init__(self, window: float = WINDOW_SECONDS) -> None:
        self.window = window
        self._lock = threading.Lock()
        self._history: deque[MetricSnapshot] = deque()
        self._start_time: float | None = None
        self._frame_count = 0
        self._fps_window: deque[float] = deque(maxlen=30)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._start_time = None
            self._frame_count = 0
            self._fps_window.clear()

    def push(self, snapshot: MetricSnapshot) -> None:
        now = snapshot.timestamp
        with self._lock:
            if self._start_time is None:
                self._start_time = now
            self._history.append(snapshot)
            self._fps_window.append(now)
            self._frame_count += 1
            cutoff = now - self.window
            while self._history and self._history[0].timestamp < cutoff:
                self._history.popleft()

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._fps_window) < 2:
                return 0.0
            span = self._fps_window[-1] - self._fps_window[0]
            if span <= 0:
                return 0.0
            return (len(self._fps_window) - 1) / span

    @property
    def latest(self) -> MetricSnapshot | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def get_timeline(self) -> TimelineData:
        with self._lock:
            if not self._history:
                return TimelineData()
            start = self._history[0].timestamp
            return TimelineData(
                timestamps=[s.timestamp - start for s in self._history],
                engagement=[s.engagement_score for s in self._history],
                eye_contact=[s.eye_contact_pct for s in self._history],
                blink_events=[1.0 if s.blink_event else 0.0 for s in self._history],
            )

    @property
    def elapsed(self) -> float:
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.time() - self._start_time


@dataclass
class TimelineData:
    """Extracted timeline arrays for plotting."""

    timestamps: list[float] = field(default_factory=list)
    engagement: list[float] = field(default_factory=list)
    eye_contact: list[float] = field(default_factory=list)
    blink_events: list[float] = field(default_factory=list)
