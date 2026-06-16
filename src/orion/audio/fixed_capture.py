from __future__ import annotations

from time import perf_counter

import numpy as np
import sounddevice as sd

from orion.audio.models import CapturedAudio
from orion.config.settings import (
    FIXED_CAPTURE_SECONDS,
    MICROPHONE_DEVICE_INDEX,
    VOICE_CHANNELS,
)


def _milliseconds(started_at: float) -> float:
    return round(
        (perf_counter() - started_at) * 1000,
        2,
    )


def capture_fixed_audio(
    device_index: int = MICROPHONE_DEVICE_INDEX,
    duration_seconds: float = FIXED_CAPTURE_SECONDS,
) -> CapturedAudio:
    timings: dict[str, float] = {}

    microphone_started_at = perf_counter()

    microphone_info = sd.query_devices(
        device_index,
        kind="input",
    )

    sample_rate = int(
        microphone_info["default_samplerate"]
    )

    sd.check_input_settings(
        device=device_index,
        samplerate=sample_rate,
        channels=VOICE_CHANNELS,
        dtype="int16",
    )

    timings["microphone_setup"] = _milliseconds(
        microphone_started_at
    )

    print(
        "ORION [MICROFONO] > Usando: "
        f"{microphone_info['name']} "
        f"(indice {device_index}, "
        f"{sample_rate} Hz)"
    )

    print(
        "ORION [AUDIO] > Grabando "
        f"{duration_seconds:.0f} segundos..."
    )

    capture_started_at = perf_counter()
    sample_count = int(
        sample_rate * duration_seconds
    )

    audio = sd.rec(
        sample_count,
        samplerate=sample_rate,
        channels=VOICE_CHANNELS,
        dtype="int16",
        device=device_index,
    )
    sd.wait()

    timings["audio_capture"] = _milliseconds(
        capture_started_at
    )

    print(
        "ORION [AUDIO] > Grabacion finalizada."
    )

    mono_audio = np.ascontiguousarray(
        audio[:, 0]
        if audio.ndim > 1
        else audio
    ).astype(np.int16)

    return CapturedAudio(
        audio_bytes=mono_audio.tobytes(),
        sample_rate=sample_rate,
        capture_mode="fixed",
        duration_ms=duration_seconds * 1000,
        timings_ms=timings,
    )
