from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from orion.wakeword.audio_stream import (
    WakeWordAudioConfig,
    WakeWordAudioStream,
    WakeWordFrameProcessor,
    float32_to_pcm16,
    pcm16_to_float32,
)
from orion.wakeword.exceptions import WakeWordAudioError


def test_callback_queues_audio_without_inference() -> None:
    stream = WakeWordAudioStream(
        config=WakeWordAudioConfig(
            queue_max_blocks=2,
        )
    )
    input_block = np.array(
        [[1], [2], [3]],
        dtype=np.int16,
    )

    stream._callback(
        input_block,
        frames=3,
        time_info=None,
        status=None,
    )
    input_block[0, 0] = 99

    queued = stream.read(
        timeout=0,
    )

    np.testing.assert_array_equal(
        queued,
        np.array(
            [1, 2, 3],
            dtype=np.int16,
        ),
    )
    assert stream.frames_received == 3


def test_bounded_queue_counts_overflow() -> None:
    stream = WakeWordAudioStream(
        config=WakeWordAudioConfig(
            queue_max_blocks=1,
        )
    )

    stream._callback(
        np.array(
            [[1]],
            dtype=np.int16,
        ),
        frames=1,
        time_info=None,
        status=None,
    )
    stream._callback(
        np.array(
            [[2]],
            dtype=np.int16,
        ),
        frames=1,
        time_info=None,
        status="overflow",
    )

    assert stream.overflow_count == 1
    assert stream.status_count == 1


def test_pcm16_to_float32_scales_once() -> None:
    samples = np.array(
        [-32768, -16384, 0, 16384, 32767],
        dtype=np.int16,
    )

    converted = pcm16_to_float32(
        samples,
    )

    np.testing.assert_allclose(
        converted,
        np.array(
            [-1.0, -0.5, 0.0, 0.5, 32767 / 32768],
            dtype=np.float32,
        ),
    )


def test_float32_to_pcm16_converts_once() -> None:
    converted = float32_to_pcm16(
        np.array(
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            dtype=np.float32,
        )
    )

    np.testing.assert_array_equal(
        converted,
        np.array(
            [-32768, -16384, 0, 16384, 32767],
            dtype=np.int16,
        ),
    )


def test_resampler_is_created_once_and_reused() -> None:
    calls = {
        "factory": 0,
        "chunks": 0,
    }

    class FakeResampler:
        def resample_chunk(
            self,
            block: np.ndarray,
            last: bool,
        ) -> np.ndarray:
            calls["chunks"] += 1
            return np.zeros(
                1280,
                dtype=np.float32,
            )

    def factory(
        native_rate: int,
        model_rate: int,
        channels: int,
        dtype: str,
    ) -> FakeResampler:
        assert native_rate == 48_000
        assert model_rate == 16_000
        assert channels == 1
        assert dtype == "float32"
        calls["factory"] += 1
        return FakeResampler()

    processor = WakeWordFrameProcessor(
        resampler_factory=factory,
    )

    processor.process_native_block(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )
    processor.process_native_block(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )

    assert calls == {
        "factory": 1,
        "chunks": 2,
    }


def test_exact_1280_sample_frames_and_partial_buffer_are_preserved() -> None:
    outputs = iter(
        [
            np.arange(
                700,
                dtype=np.float32,
            )
            / 10000,
            np.arange(
                700,
                1400,
                dtype=np.float32,
            )
            / 10000,
        ]
    )

    class FakeResampler:
        def resample_chunk(
            self,
            block: np.ndarray,
            last: bool,
        ) -> np.ndarray:
            return next(
                outputs,
            )

    processor = WakeWordFrameProcessor(
        resampler_factory=lambda *args, **kwargs: FakeResampler(),
    )

    first = processor.process_native_block(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )
    second = processor.process_native_block(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )

    assert first == []
    assert len(second) == 1
    assert second[0].shape == (1280,)
    assert processor.pending_model_frames == 120
    assert processor.model_frames_produced == 1400
    assert processor.inference_blocks == 1


def test_audio_stream_start_and_stop_release_stream() -> None:
    lifecycle: list[str] = []

    class FakeInputStream:
        def __init__(
            self,
            **kwargs: object,
        ) -> None:
            assert kwargs["samplerate"] == 48_000
            assert kwargs["channels"] == 1
            assert kwargs["dtype"] == "int16"
            assert callable(kwargs["callback"])

        def start(
            self,
        ) -> None:
            lifecycle.append(
                "start",
            )

        def stop(
            self,
        ) -> None:
            lifecycle.append(
                "stop",
            )

        def close(
            self,
        ) -> None:
            lifecycle.append(
                "close",
            )

    fake_sounddevice = SimpleNamespace(
        check_input_settings=lambda **kwargs: None,
        InputStream=FakeInputStream,
    )
    stream = WakeWordAudioStream(
        sounddevice_module=fake_sounddevice,
    )

    stream.start()
    stream.stop()

    assert lifecycle == [
        "start",
        "stop",
        "close",
    ]


def test_audio_stream_start_wraps_microphone_errors() -> None:
    fake_sounddevice = SimpleNamespace(
        check_input_settings=lambda **kwargs: (_raise_runtime_error()),
    )
    stream = WakeWordAudioStream(
        sounddevice_module=fake_sounddevice,
    )

    with pytest.raises(WakeWordAudioError):
        stream.start()


def _raise_runtime_error() -> None:
    raise RuntimeError(
        "mic unavailable",
    )
