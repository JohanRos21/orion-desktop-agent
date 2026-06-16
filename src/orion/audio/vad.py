from __future__ import annotations

from orion.audio.vad_capture import (
    SpeechNotDetectedError,
    capture_vad_audio,
    get_vad_model,
)


capture_speech = capture_vad_audio

__all__ = [
    "SpeechNotDetectedError",
    "capture_speech",
    "capture_vad_audio",
    "get_vad_model",
]
