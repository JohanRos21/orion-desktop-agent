from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WakeWordPrediction:
    model_name: str
    score: float
    detected: bool
    timestamp: float


@dataclass(frozen=True, slots=True)
class WakeWordEvent:
    wake_word: str
    score: float
    detected_at: float


@dataclass(frozen=True, slots=True)
class WakeWordDebugStats:
    native_sample_rate: int
    model_sample_rate: int
    native_frames_received: int
    model_frames_produced: int
    inference_blocks: int
    max_score: float
    average_score: float
    overflows: int
    status_count: int
    total_ms: float
