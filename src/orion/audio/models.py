from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CapturedAudio:
    audio_bytes: bytes
    sample_rate: int
    capture_mode: str
    duration_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)
