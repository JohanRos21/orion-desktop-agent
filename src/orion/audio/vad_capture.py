from __future__ import annotations

from collections import deque
from functools import lru_cache
from math import ceil
from time import perf_counter
from typing import Any

from orion.audio.models import CapturedAudio
from orion.config.settings import (
    MICROPHONE_DEVICE_INDEX,
    VOICE_SAMPLE_WIDTH_BYTES,
)


VAD_SAMPLE_RATE = 16_000
VAD_CHUNK_SAMPLES = 512

VAD_THRESHOLD = 0.08
VAD_EXIT_THRESHOLD = 0.05

PRE_ROLL_MS = 400
START_CONFIRMATION_MS = 160
END_SILENCE_MS = 1_000
TRAILING_SILENCE_MS = 250

WAIT_TIMEOUT_SECONDS = 12.0
MAX_SPEECH_SECONDS = 30.0
MIN_SPEECH_SECONDS = 0.80


class SpeechNotDetectedError(RuntimeError):
    """No se detecto voz antes del tiempo limite."""


def _milliseconds(started_at: float) -> float:
    return round(
        (perf_counter() - started_at) * 1000,
        2,
    )


@lru_cache(maxsize=1)
def get_vad_model() -> Any:
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=True)


def _audio_levels(samples: Any) -> tuple[float, float]:
    import numpy as np

    if samples.size == 0:
        return 0.0, 0.0

    rms = float(
        np.sqrt(
            np.mean(samples * samples)
        )
    )

    peak = float(
        np.max(np.abs(samples))
    )

    return rms, peak


def _convert_native_audio_to_vad(
    native_samples: Any,
    native_sample_rate: int,
) -> Any:
    import numpy as np

    if native_sample_rate % VAD_SAMPLE_RATE != 0:
        raise ValueError(
            "La frecuencia del microfono debe ser multiplo "
            f"de {VAD_SAMPLE_RATE} Hz."
        )

    reduction_factor = (
        native_sample_rate // VAD_SAMPLE_RATE
    )

    expected_native_samples = (
        VAD_CHUNK_SAMPLES * reduction_factor
    )

    if native_samples.size != expected_native_samples:
        raise ValueError(
            "El fragmento de audio recibido tiene un tamano "
            "inesperado."
        )

    float_samples = native_samples.astype(
        np.float32
    )
    downsampled = float_samples[::reduction_factor]

    if downsampled.size != VAD_CHUNK_SAMPLES:
        raise ValueError(
            "La conversion a 16 kHz produjo un tamano inesperado."
        )

    return np.clip(
        downsampled / 32768.0,
        -1.0,
        1.0,
    )


def _duration_ms(
    audio_bytes: bytes,
    sample_rate: int,
) -> float:
    return round(
        len(audio_bytes)
        / (sample_rate * VOICE_SAMPLE_WIDTH_BYTES)
        * 1000,
        2,
    )


def capture_vad_audio(
    device_index: int = MICROPHONE_DEVICE_INDEX,
) -> CapturedAudio:
    import numpy as np
    import sounddevice as sd
    import torch

    total_started_at = perf_counter()
    timings: dict[str, float] = {}

    vad_started_at = perf_counter()
    vad_model = get_vad_model()
    vad_model.reset_states()

    timings["vad_model_ready"] = _milliseconds(
        vad_started_at
    )

    microphone_started_at = perf_counter()

    microphone_info = sd.query_devices(
        device_index,
        kind="input",
    )

    native_sample_rate = int(
        microphone_info["default_samplerate"]
    )

    if native_sample_rate % VAD_SAMPLE_RATE != 0:
        raise ValueError(
            "Silero VAD necesita que la frecuencia del "
            "microfono sea multiplo de 16 kHz. "
            f"Frecuencia detectada: {native_sample_rate} Hz."
        )

    reduction_factor = (
        native_sample_rate // VAD_SAMPLE_RATE
    )

    native_chunk_samples = (
        VAD_CHUNK_SAMPLES * reduction_factor
    )

    sd.check_input_settings(
        device=device_index,
        samplerate=native_sample_rate,
        channels=1,
        dtype="int16",
    )

    timings["microphone_setup"] = _milliseconds(
        microphone_started_at
    )

    chunk_duration_ms = (
        VAD_CHUNK_SAMPLES
        / VAD_SAMPLE_RATE
        * 1000
    )

    pre_roll_chunks = max(
        1,
        ceil(PRE_ROLL_MS / chunk_duration_ms),
    )

    start_confirmation_chunks = max(
        1,
        ceil(
            START_CONFIRMATION_MS
            / chunk_duration_ms
        ),
    )

    end_silence_chunks = max(
        1,
        ceil(
            END_SILENCE_MS
            / chunk_duration_ms
        ),
    )

    trailing_silence_chunks = max(
        1,
        ceil(
            TRAILING_SILENCE_MS
            / chunk_duration_ms
        ),
    )

    pre_roll: deque[bytes] = deque(
        maxlen=pre_roll_chunks
    )

    captured_chunks: list[bytes] = []

    speech_started = False
    speech_started_at: float | None = None

    speech_start_streak = 0
    silence_streak = 0

    max_speech_probability = 0.0
    max_audio_rms = 0.0
    max_audio_peak = 0.0

    print(
        "ORION [MICROFONO] > Usando: "
        f"{microphone_info['name']} "
        f"(indice {device_index}, "
        f"{native_sample_rate} Hz)"
    )

    print(
        "ORION [VAD] > Modo experimental. Esperando voz..."
    )

    waiting_started_at = perf_counter()

    with sd.InputStream(
        device=device_index,
        samplerate=native_sample_rate,
        blocksize=native_chunk_samples,
        channels=1,
        dtype="int16",
    ) as stream:
        while True:
            audio_chunk, overflowed = stream.read(
                native_chunk_samples
            )

            if overflowed:
                print(
                    "ORION [AVISO] > "
                    "El microfono perdio un fragmento de audio."
                )

            native_samples = (
                audio_chunk[:, 0]
                .copy()
                .astype(np.int16)
            )

            raw_audio = native_samples.tobytes()

            vad_samples = _convert_native_audio_to_vad(
                native_samples=native_samples,
                native_sample_rate=native_sample_rate,
            )

            vad_tensor = torch.from_numpy(
                vad_samples
            )

            with torch.inference_mode():
                speech_probability = float(
                    vad_model(
                        vad_tensor,
                        VAD_SAMPLE_RATE,
                    ).item()
                )

            audio_rms, audio_peak = _audio_levels(
                vad_samples
            )

            max_speech_probability = max(
                max_speech_probability,
                speech_probability,
            )
            max_audio_rms = max(
                max_audio_rms,
                audio_rms,
            )
            max_audio_peak = max(
                max_audio_peak,
                audio_peak,
            )

            if not speech_started:
                pre_roll.append(raw_audio)

                if speech_probability >= VAD_THRESHOLD:
                    speech_start_streak += 1
                else:
                    speech_start_streak = 0

                if (
                    speech_start_streak
                    >= start_confirmation_chunks
                ):
                    speech_started = True
                    speech_started_at = perf_counter()

                    timings["waiting_for_speech"] = (
                        _milliseconds(
                            waiting_started_at
                        )
                    )

                    captured_chunks.extend(
                        pre_roll
                    )

                    pre_roll.clear()

                    print(
                        "ORION [VAD] > Voz detectada."
                    )

                    continue

                waiting_seconds = (
                    perf_counter()
                    - waiting_started_at
                )

                if waiting_seconds >= WAIT_TIMEOUT_SECONDS:
                    raise SpeechNotDetectedError(
                        "No detecte que comenzaras a hablar. "
                        "Senal maxima del microfono: "
                        f"{max_audio_rms * 100:.1f}% "
                        f"(pico {max_audio_peak * 100:.1f}%). "
                        "VAD maximo: "
                        f"{max_speech_probability:.2f}."
                    )

                continue

            captured_chunks.append(raw_audio)

            if speech_probability < VAD_EXIT_THRESHOLD:
                silence_streak += 1
            else:
                silence_streak = 0

            speech_duration_seconds = (
                perf_counter() - speech_started_at
                if speech_started_at is not None
                else 0.0
            )

            if (
                speech_duration_seconds >= MIN_SPEECH_SECONDS
                and silence_streak >= end_silence_chunks
            ):
                removable_silence = max(
                    0,
                    silence_streak
                    - trailing_silence_chunks,
                )

                if removable_silence:
                    del captured_chunks[
                        -removable_silence:
                    ]

                print(
                    "ORION [VAD] > Fin de voz detectado."
                )

                break

            if (
                speech_started_at is not None
                and speech_duration_seconds >= MAX_SPEECH_SECONDS
            ):
                print(
                    "ORION [VAD] > Se alcanzo el maximo de "
                    f"{MAX_SPEECH_SECONDS:.0f} segundos."
                )

                break

    if speech_started_at is None:
        raise SpeechNotDetectedError(
            "No se detecto una instruccion hablada."
        )

    timings["speech_capture"] = _milliseconds(
        speech_started_at
    )

    timings["vad_total"] = _milliseconds(
        total_started_at
    )

    audio_bytes = b"".join(captured_chunks)

    return CapturedAudio(
        audio_bytes=audio_bytes,
        sample_rate=native_sample_rate,
        capture_mode="vad",
        duration_ms=_duration_ms(
            audio_bytes=audio_bytes,
            sample_rate=native_sample_rate,
        ),
        timings_ms=timings,
    )
