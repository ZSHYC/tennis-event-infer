from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryFrame:
    frame_number: int
    detected: bool
    x: float | None
    y: float | None
    width: int
    height: int


@dataclass(frozen=True)
class Event:
    frame_number: int
    timestamp_seconds: float
    event_type: str
    confidence: float
    x: float | None
    y: float | None
