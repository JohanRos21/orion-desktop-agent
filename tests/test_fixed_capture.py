from types import SimpleNamespace

import numpy as np
import pytest

from orion.audio import fixed_capture


def test_capture_fixed_audio_uses_mocked_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec_calls: dict[str, object] = {}
    waited = {"value": False}

    def query_devices(
        device_index: int,
        kind: str,
    ) -> dict[str, object]:
        assert device_index == 15
        assert kind == "input"
        return {
            "name": "Mock Microphone",
            "default_samplerate": 48_000,
        }

    def check_input_settings(**kwargs: object) -> None:
        assert kwargs == {
            "device": 15,
            "samplerate": 48_000,
            "channels": 1,
            "dtype": "int16",
        }

    def rec(
        sample_count: int,
        **kwargs: object,
    ) -> np.ndarray:
        rec_calls["sample_count"] = sample_count
        rec_calls.update(kwargs)
        return np.array(
            [[1], [2], [3]],
            dtype=np.int16,
        )

    def wait() -> None:
        waited["value"] = True

    monkeypatch.setattr(
        fixed_capture,
        "sd",
        SimpleNamespace(
            query_devices=query_devices,
            check_input_settings=check_input_settings,
            rec=rec,
            wait=wait,
        ),
    )

    captured_audio = fixed_capture.capture_fixed_audio(
        device_index=15,
        duration_seconds=5,
    )

    assert rec_calls == {
        "sample_count": 240_000,
        "samplerate": 48_000,
        "channels": 1,
        "dtype": "int16",
        "device": 15,
    }
    assert waited["value"] is True
    assert captured_audio.audio_bytes == (
        np.array([1, 2, 3], dtype=np.int16)
        .tobytes()
    )
    assert captured_audio.sample_rate == 48_000
    assert captured_audio.capture_mode == "fixed"
    assert captured_audio.duration_ms == 5_000
    assert "microphone_setup" in captured_audio.timings_ms
    assert "audio_capture" in captured_audio.timings_ms
