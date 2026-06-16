from dataclasses import FrozenInstanceError

import pytest

from orion.audio.models import CapturedAudio


def test_captured_audio_structure_is_immutable() -> None:
    captured_audio = CapturedAudio(
        audio_bytes=b"abc",
        sample_rate=48_000,
        capture_mode="fixed",
        duration_ms=5_000.0,
        timings_ms={"audio_capture": 5_000.0},
    )

    assert captured_audio.audio_bytes == b"abc"
    assert captured_audio.sample_rate == 48_000
    assert captured_audio.capture_mode == "fixed"
    assert captured_audio.duration_ms == 5_000.0
    assert captured_audio.timings_ms["audio_capture"] == 5_000.0

    with pytest.raises(FrozenInstanceError):
        captured_audio.sample_rate = 16_000
