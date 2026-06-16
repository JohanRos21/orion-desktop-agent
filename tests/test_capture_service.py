from types import ModuleType
import sys

import pytest

from orion.audio.capture_service import (
    UnsupportedCaptureModeError,
    capture_audio,
)
from orion.audio.models import CapturedAudio
from orion.config import settings


def test_capture_audio_selects_fixed_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_fixed_capture = ModuleType(
        "orion.audio.fixed_capture"
    )
    calls: dict[str, float] = {}

    def capture_fixed_audio(
        duration_seconds: float,
    ) -> CapturedAudio:
        calls["duration_seconds"] = duration_seconds
        return CapturedAudio(
            audio_bytes=b"fixed",
            sample_rate=48_000,
            capture_mode="fixed",
            duration_ms=duration_seconds * 1000,
            timings_ms={"audio_capture": 1.0},
        )

    fake_fixed_capture.capture_fixed_audio = capture_fixed_audio

    monkeypatch.setattr(
        settings,
        "VOICE_CAPTURE_MODE",
        "fixed",
    )
    monkeypatch.setattr(
        settings,
        "FIXED_CAPTURE_SECONDS",
        5,
    )
    monkeypatch.setitem(
        sys.modules,
        "orion.audio.fixed_capture",
        fake_fixed_capture,
    )
    monkeypatch.delitem(
        sys.modules,
        "orion.audio.vad_capture",
        raising=False,
    )

    captured_audio = capture_audio()

    assert captured_audio.capture_mode == "fixed"
    assert captured_audio.audio_bytes == b"fixed"
    assert calls["duration_seconds"] == 5
    assert "orion.audio.vad_capture" not in sys.modules


def test_capture_audio_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "VOICE_CAPTURE_MODE",
        "unknown",
    )

    with pytest.raises(UnsupportedCaptureModeError):
        capture_audio()
