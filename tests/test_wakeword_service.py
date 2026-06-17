from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orion.wakeword.exceptions import (
    UnsupportedWakeWordModelError,
    WakeWordModelError,
)
from orion.wakeword.models import WakeWordEvent
from orion.wakeword.service import (
    OpenWakeWordModelAdapter,
    WakeWordModelReference,
    WakeWordService,
    resolve_model_reference,
)


def test_model_is_loaded_once() -> None:
    loads: list[str] = []

    def loader(
        reference: WakeWordModelReference,
    ) -> object:
        loads.append(
            reference.model_ref,
        )
        return _FakeOpenWakeWordModel(
            [{"hey jarvis": 0.1}],
        )

    adapter = OpenWakeWordModelAdapter(
        model_loader=loader,
    )

    adapter.load()
    adapter.load()

    assert loads == ["hey jarvis"]


def test_threshold_below_does_not_detect() -> None:
    service = _service_with_scores(
        [0.49],
    )

    events = service.process_audio(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )

    assert events == []


def test_threshold_reached_detects() -> None:
    service = _service_with_scores(
        [0.50],
    )

    events = service.process_audio(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )

    assert events == [
        WakeWordEvent(
            wake_word="hey jarvis",
            score=0.50,
            detected_at=0.0,
        )
    ]


def test_cooldown_prevents_duplicate_events() -> None:
    service = _service_with_scores(
        [0.90, 0.91],
        time_values=[0.0, 1.0],
        cooldown_seconds=2.0,
    )

    first = service.process_audio(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )
    second = service.process_audio(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )

    assert len(first) == 1
    assert second == []


def test_detection_after_cooldown_can_fire_again() -> None:
    service = _service_with_scores(
        [0.90, 0.91],
        time_values=[0.0, 2.1],
        cooldown_seconds=2.0,
    )

    first = service.process_audio(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )
    second = service.process_audio(
        np.zeros(
            3840,
            dtype=np.int16,
        )
    )

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].detected_at == 2.1


def test_stop_releases_stream() -> None:
    stream = _FakeAudioStream()
    service = _service_with_scores(
        [0.0],
        audio_stream=stream,
    )

    service.start()
    service.stop()

    assert stream.started is True
    assert stream.stopped is True


def test_missing_model_path_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(
        WakeWordModelError,
        match="No existe",
    ):
        resolve_model_reference(
            tmp_path / "missing.onnx",
        )


def test_tflite_model_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "bad.tflite"
    model_path.write_bytes(
        b"tflite",
    )

    with pytest.raises(UnsupportedWakeWordModelError):
        resolve_model_reference(
            model_path,
        )


def test_onnx_model_path_is_accepted(tmp_path: Path) -> None:
    model_path = tmp_path / "orion.onnx"
    model_path.write_bytes(
        b"onnx",
    )

    reference = resolve_model_reference(
        model_path,
    )

    assert reference.model_ref == str(model_path)
    assert reference.display_name == "orion"


class _FakeOpenWakeWordModel:
    def __init__(
        self,
        predictions: list[dict[str, float]],
    ) -> None:
        self.predictions = iter(
            predictions,
        )

    def predict(
        self,
        frame: np.ndarray,
    ) -> dict[str, float]:
        assert frame.dtype == np.int16
        assert frame.shape == (1280,)
        return next(
            self.predictions,
        )


class _FakeModelAdapter:
    display_name = "hey jarvis"

    def __init__(
        self,
        scores: list[float],
    ) -> None:
        self.model = _FakeOpenWakeWordModel(
            [
                {
                    "hey jarvis": score,
                }
                for score in scores
            ]
        )
        self.load_count = 0

    def load(
        self,
    ) -> object:
        self.load_count += 1
        return self.model

    def predict(
        self,
        frame: np.ndarray,
    ) -> dict[str, float]:
        return self.model.predict(
            frame,
        )


class _FakeFrameProcessor:
    native_frames_received = 0
    model_frames_produced = 0
    inference_blocks = 0

    def process_native_block(
        self,
        native_block: np.ndarray,
    ) -> list[np.ndarray]:
        self.native_frames_received += int(
            native_block.size,
        )
        self.model_frames_produced += 1280
        self.inference_blocks += 1
        return [
            np.zeros(
                1280,
                dtype=np.int16,
            )
        ]


class _FakeAudioStream:
    overflow_count = 0
    status_count = 0

    def __init__(
        self,
    ) -> None:
        self.started = False
        self.stopped = False

    def start(
        self,
    ) -> None:
        self.started = True

    def stop(
        self,
    ) -> None:
        self.stopped = True

    def read(
        self,
        timeout: float,
    ) -> np.ndarray | None:
        return None


def _service_with_scores(
    scores: list[float],
    time_values: list[float] | None = None,
    cooldown_seconds: float = 2.0,
    audio_stream: _FakeAudioStream | None = None,
) -> WakeWordService:
    times = iter(
        time_values or [0.0] * len(scores),
    )

    return WakeWordService(
        threshold=0.50,
        cooldown_seconds=cooldown_seconds,
        model_adapter=_FakeModelAdapter(
            scores,
        ),
        audio_stream=audio_stream or _FakeAudioStream(),
        frame_processor=_FakeFrameProcessor(),
        time_func=lambda: next(
            times,
        ),
    )
