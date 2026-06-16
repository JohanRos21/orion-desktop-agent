from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any
import unicodedata
import wave

import numpy as np

from orion.audio.capture_service import (
    UnsupportedCaptureModeError,
    capture_audio,
)
from orion.config.settings import (
    VOICE_CHANNELS,
    VOICE_SAMPLE_WIDTH_BYTES,
)
from orion.runtime.nvidia import configure_nvidia_runtime

# Debe ejecutarse antes de importar faster-whisper/CTranslate2.
NVIDIA_DLL_DIRECTORIES = configure_nvidia_runtime()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_DIR = PROJECT_ROOT / "models"
LAST_CAPTURE_PATH = PROJECT_ROOT / "last-voice-capture.wav"

MODEL_SIZE = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

MIN_TRANSCRIPTION_RMS = 0.003
MIN_TRANSCRIPTION_PEAK = 0.015

HALLUCINATED_TRANSCRIPTS = (
    "subtitulos realizados por la comunidad de amara org",
    "subtitulos por la comunidad de amara org",
    "subtitulos creados por la comunidad de amara org",
    "amara org",
    "gracias por ver el video",
    "suscribete al canal",
    "thank you for watching",
)


@dataclass(frozen=True, slots=True)
class VoiceResult:
    success: bool
    message: str
    transcript: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


def _milliseconds(start_time: float) -> float:
    return round(
        (perf_counter() - start_time) * 1000,
        2,
    )


def _audio_levels(
    audio_bytes: bytes,
) -> tuple[float, float]:
    samples = np.frombuffer(
        audio_bytes,
        dtype=np.int16,
    )

    if samples.size == 0:
        return 0.0, 0.0

    float_samples = (
        samples.astype(np.float32)
        / 32768.0
    )

    rms = float(
        np.sqrt(
            np.mean(float_samples * float_samples)
        )
    )

    peak = float(
        np.max(np.abs(float_samples))
    )

    return rms, peak


def _normalize_transcript(
    text: str,
) -> str:
    decomposed_text = unicodedata.normalize(
        "NFD",
        text.casefold().strip(),
    )

    without_accents = "".join(
        character
        for character in decomposed_text
        if unicodedata.category(character) != "Mn"
    )

    without_punctuation = re.sub(
        r"[^a-z0-9]+",
        " ",
        without_accents,
    )

    return " ".join(
        without_punctuation.split()
    )


def _looks_hallucinated(
    transcript: str,
) -> bool:
    normalized_transcript = _normalize_transcript(
        transcript
    )

    return any(
        phrase in normalized_transcript
        for phrase in HALLUCINATED_TRANSCRIPTS
    )


@lru_cache(maxsize=1)
def get_whisper_model() -> Any:
    from faster_whisper import WhisperModel

    MODEL_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_options = {
        "model_size_or_path": MODEL_SIZE,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "download_root": str(MODEL_CACHE_DIR),
    }

    model_cache_marker = (
        MODEL_CACHE_DIR
        / f"models--Systran--faster-whisper-{MODEL_SIZE}"
    )

    if model_cache_marker.exists():
        return WhisperModel(
            **model_options,
            local_files_only=True,
        )

    print(
        "ORION [MODELO] > El modelo local todavia no existe. "
        f"Descargando faster-whisper {MODEL_SIZE}..."
    )

    return WhisperModel(
        **model_options,
        local_files_only=False,
    )


def _save_wav(
    audio_bytes: bytes,
    output_path: Path,
    sample_rate: int,
) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(VOICE_CHANNELS)
        wav_file.setsampwidth(VOICE_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)


def listen_once() -> VoiceResult:
    total_started_at = perf_counter()
    temporary_path: Path | None = None
    timings: dict[str, float] = {}

    try:
        captured_audio = capture_audio()

        timings.update(
            captured_audio.timings_ms
        )

        audio_rms, audio_peak = _audio_levels(
            captured_audio.audio_bytes
        )

        if (
            audio_rms < MIN_TRANSCRIPTION_RMS
            or audio_peak < MIN_TRANSCRIPTION_PEAK
        ):
            timings["total"] = _milliseconds(
                total_started_at
            )

            return VoiceResult(
                success=False,
                message=(
                    "El microfono capturo una senal muy baja. "
                    "Repite el comando mas cerca del microfono."
                ),
                timings_ms=timings,
            )

        save_started_at = perf_counter()

        with NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

        _save_wav(
            audio_bytes=captured_audio.audio_bytes,
            output_path=temporary_path,
            sample_rate=captured_audio.sample_rate,
        )

        _save_wav(
            audio_bytes=captured_audio.audio_bytes,
            output_path=LAST_CAPTURE_PATH,
            sample_rate=captured_audio.sample_rate,
        )

        timings["save_audio"] = _milliseconds(
            save_started_at
        )

        model_started_at = perf_counter()
        model = get_whisper_model()

        timings["model_ready"] = _milliseconds(
            model_started_at
        )

        transcription_started_at = perf_counter()

        segments, _ = model.transcribe(
            str(temporary_path),
            language="es",
            beam_size=5,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=(
                "Asistente ORION para controlar Windows. "
                "Aplicaciones: calculadora, bloc de notas, "
                "explorador de archivos, Visual Studio Code, "
                "Discord y navegador."
            ),
        )

        segment_list = list(segments)

        transcript = " ".join(
            segment.text.strip()
            for segment in segment_list
            if segment.text.strip()
        ).strip()

        timings["transcription"] = _milliseconds(
            transcription_started_at
        )

        timings["total"] = _milliseconds(
            total_started_at
        )

        if not transcript:
            return VoiceResult(
                success=False,
                message=(
                    "Escuche tu voz, pero no pude "
                    "transcribir la instruccion."
                ),
                timings_ms=timings,
            )

        if _looks_hallucinated(transcript):
            return VoiceResult(
                success=False,
                message=(
                    "El audio no fue lo bastante claro para "
                    "entender el comando. Repite la frase mas "
                    "cerca del microfono."
                ),
                timings_ms=timings,
            )

        return VoiceResult(
            success=True,
            message="Comando reconocido.",
            transcript=transcript,
            timings_ms=timings,
        )

    except UnsupportedCaptureModeError as error:
        timings["total"] = _milliseconds(
            total_started_at
        )

        return VoiceResult(
            success=False,
            message=str(error),
            timings_ms=timings,
        )

    except OSError as error:
        timings["total"] = _milliseconds(
            total_started_at
        )

        return VoiceResult(
            success=False,
            message=(
                f"No pude procesar el audio: {error}"
            ),
            timings_ms=timings,
        )

    except Exception as error:
        timings["total"] = _milliseconds(
            total_started_at
        )

        if error.__class__.__name__ == "PortAudioError":
            message = (
                f"No pude acceder al microfono: {error}"
            )
        else:
            message = (
                f"Fallo la transcripcion local: {error}"
            )

        return VoiceResult(
            success=False,
            message=message,
            timings_ms=timings,
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )
