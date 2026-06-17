from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

from orion.config import settings
from orion.wakeword.exceptions import (
    WakeWordAudioError,
    WakeWordDependencyError,
)


PCM16_MAX_ABS = 32768.0


@dataclass(frozen=True, slots=True)
class WakeWordAudioConfig:
    device_index: int = settings.MICROPHONE_DEVICE_INDEX
    native_sample_rate: int = settings.WAKEWORD_NATIVE_SAMPLE_RATE
    model_sample_rate: int = settings.WAKEWORD_MODEL_SAMPLE_RATE
    channels: int = settings.WAKEWORD_CHANNELS
    frame_ms: int = settings.WAKEWORD_FRAME_MS
    queue_max_blocks: int = 32
    dtype: str = "int16"

    @property
    def model_frame_samples(
        self,
    ) -> int:
        return int(
            self.model_sample_rate
            * self.frame_ms
            / 1000
        )

    @property
    def native_block_samples(
        self,
    ) -> int:
        return int(
            self.native_sample_rate
            * self.frame_ms
            / 1000
        )


class WakeWordAudioStream:
    def __init__(
        self,
        config: WakeWordAudioConfig | None = None,
        sounddevice_module: Any | None = None,
    ) -> None:
        self.config = config or WakeWordAudioConfig()
        self._sounddevice_module = sounddevice_module
        self._queue: Queue[np.ndarray] = Queue(
            maxsize=self.config.queue_max_blocks,
        )
        self._stream: Any | None = None
        self.overflow_count = 0
        self.status_count = 0
        self.frames_received = 0

    def start(
        self,
    ) -> None:
        if self._stream is not None:
            return

        sd = self._sounddevice_module or _import_sounddevice()

        try:
            sd.check_input_settings(
                device=self.config.device_index,
                samplerate=self.config.native_sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
            )
            self._stream = sd.InputStream(
                device=self.config.device_index,
                samplerate=self.config.native_sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.native_block_samples,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as error:
            self._stream = None
            raise WakeWordAudioError(
                f"No pude abrir el microfono para wake word: {error}"
            ) from error

    def stop(
        self,
    ) -> None:
        if self._stream is None:
            return

        stream = self._stream
        self._stream = None

        if hasattr(
            stream,
            "stop",
        ):
            stream.stop()

        if hasattr(
            stream,
            "close",
        ):
            stream.close()

    def read(
        self,
        timeout: float = 0.1,
    ) -> np.ndarray | None:
        try:
            return self._queue.get(
                timeout=timeout,
            )
        except Empty:
            return None

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if status:
            self.status_count += 1

        block = _mono_pcm16_copy(
            indata,
        )

        try:
            self._queue.put_nowait(
                block,
            )
            self.frames_received += int(
                frames,
            )
        except Full:
            self.overflow_count += 1


class WakeWordFrameProcessor:
    def __init__(
        self,
        config: WakeWordAudioConfig | None = None,
        resampler_factory: Any | None = None,
    ) -> None:
        self.config = config or WakeWordAudioConfig()
        self.resampler = _create_resampler(
            config=self.config,
            resampler_factory=resampler_factory,
        )
        self._buffer = np.array(
            [],
            dtype=np.float32,
        )
        self.native_frames_received = 0
        self.model_frames_produced = 0
        self.inference_blocks = 0
        self.pcm16_conversions = 0

    def process_native_block(
        self,
        native_block: np.ndarray,
    ) -> list[np.ndarray]:
        mono_pcm16 = _mono_pcm16_copy(
            native_block,
        )
        self.native_frames_received += int(
            mono_pcm16.size,
        )
        float_block = pcm16_to_float32(
            mono_pcm16,
        )

        model_float = np.asarray(
            self.resampler.resample_chunk(
                float_block,
                last=False,
            ),
            dtype=np.float32,
        )
        self.model_frames_produced += int(
            model_float.size,
        )

        if self._buffer.size:
            self._buffer = np.concatenate(
                [
                    self._buffer,
                    model_float,
                ]
            )
        else:
            self._buffer = model_float

        frames: list[np.ndarray] = []
        frame_samples = self.config.model_frame_samples

        while self._buffer.size >= frame_samples:
            frame_float = np.ascontiguousarray(
                self._buffer[:frame_samples],
                dtype=np.float32,
            )
            self._buffer = self._buffer[
                frame_samples:
            ]
            frames.append(
                float32_to_pcm16(
                    frame_float,
                )
            )
            self.inference_blocks += 1
            self.pcm16_conversions += 1

        return frames

    @property
    def pending_model_frames(
        self,
    ) -> int:
        return int(
            self._buffer.size,
        )


def pcm16_to_float32(
    samples: np.ndarray,
) -> np.ndarray:
    return (
        _mono_pcm16_copy(
            samples,
        ).astype(
            np.float32,
        )
        / PCM16_MAX_ABS
    )


def float32_to_pcm16(
    samples: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        np.asarray(
            samples,
            dtype=np.float32,
        ),
        -1.0,
        32767.0 / PCM16_MAX_ABS,
    )

    return np.ascontiguousarray(
        (
            clipped
            * PCM16_MAX_ABS
        ).astype(
            np.int16,
        )
    )


def _mono_pcm16_copy(
    samples: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        samples,
        dtype=np.int16,
    )

    if array.ndim > 1:
        array = array[
            :,
            0,
        ]

    return np.ascontiguousarray(
        array,
        dtype=np.int16,
    ).copy()


def _create_resampler(
    config: WakeWordAudioConfig,
    resampler_factory: Any | None,
) -> Any:
    if resampler_factory is not None:
        return resampler_factory(
            config.native_sample_rate,
            config.model_sample_rate,
            config.channels,
            dtype="float32",
        )

    try:
        import soxr
    except ImportError as error:
        raise WakeWordDependencyError(
            "Falta la dependencia 'soxr'. Instala las dependencias "
            "del proyecto antes de iniciar wake word."
        ) from error

    return soxr.ResampleStream(
        config.native_sample_rate,
        config.model_sample_rate,
        config.channels,
        dtype="float32",
    )


def _import_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError as error:
        raise WakeWordDependencyError(
            "Falta la dependencia 'sounddevice'."
        ) from error

    return sd
