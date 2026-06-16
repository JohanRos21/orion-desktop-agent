from types import SimpleNamespace

import numpy as np
import pytest

from orion.audio.models import CapturedAudio
from orion import voice


def test_listen_once_uses_capture_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    audio_samples = np.full(
        48_000,
        2_000,
        dtype=np.int16,
    )

    captured_audio = CapturedAudio(
        audio_bytes=audio_samples.tobytes(),
        sample_rate=48_000,
        capture_mode="fixed",
        duration_ms=1_000.0,
        timings_ms={"audio_capture": 1_000.0},
    )

    class FakeWhisperModel:
        def transcribe(
            self,
            path: str,
            **options: object,
        ) -> tuple[list[SimpleNamespace], object]:
            assert path.endswith(".wav")
            assert options["language"] == "es"
            assert options["vad_filter"] is False

            return (
                [
                    SimpleNamespace(
                        text=" Abre la calculadora"
                    )
                ],
                object(),
            )

    monkeypatch.setattr(
        voice,
        "LAST_CAPTURE_PATH",
        tmp_path / "last-voice-capture.wav",
    )
    monkeypatch.setattr(
        voice,
        "capture_audio",
        lambda: captured_audio,
    )
    monkeypatch.setattr(
        voice,
        "get_whisper_model",
        lambda: FakeWhisperModel(),
    )

    result = voice.listen_once()

    assert result.success is True
    assert result.transcript == "Abre la calculadora"
    assert result.timings_ms["audio_capture"] == 1_000.0
    assert "model_ready" in result.timings_ms
    assert "transcription" in result.timings_ms
    assert (tmp_path / "last-voice-capture.wav").exists()
