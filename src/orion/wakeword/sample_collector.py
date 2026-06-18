from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil, log10, sqrt
from pathlib import Path
import wave

import numpy as np

from orion.config import settings
from orion.wakeword.audio_stream import (
    WakeWordAudioConfig,
    WakeWordFrameProcessor,
)
from orion.wakeword.exceptions import (
    WakeWordAudioError,
    WakeWordError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "local_models"
    / "wakeword"
    / "dataset"
)
MANIFEST_FILENAME = "manifest.json"
POSITIVE_COUNT = 20
NEGATIVE_CLIP_SECONDS = 2.0
NEGATIVE_TOTAL_SECONDS = 60.0
MIN_CLIP_SECONDS = 0.25

POSITIVE_INSTRUCTIONS = (
    "voz normal",
    "voz mas baja",
    "voz mas alta",
    "pronunciacion rapida",
    "pronunciacion lenta",
    "cerca del microfono",
    "a aproximadamente un metro",
    "con ruido ambiental normal",
)
NEGATIVE_INSTRUCTION = (
    "No pronuncies Orion durante esta grabacion."
)

InputFunction = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class AudioMetrics:
    duration_seconds: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    rms_dbfs: float
    peak: float
    rms: float


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    tipo: str
    ruta_relativa: str
    duracion: float
    sample_rate: int
    canales: int
    peak_dbfs: float
    rms_dbfs: float
    fecha: str
    instruccion: str


def collect_positive_samples(
    count: int = POSITIVE_COUNT,
    duration_seconds: float = NEGATIVE_CLIP_SECONDS,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    overwrite: bool = False,
    input_func: InputFunction = input,
    recorder: Callable[[float], np.ndarray] | None = None,
) -> list[ManifestEntry]:
    return _collect_samples(
        kind="positive",
        count=count,
        duration_seconds=duration_seconds,
        dataset_root=dataset_root,
        overwrite=overwrite,
        input_func=input_func,
        recorder=recorder,
    )


def collect_negative_samples(
    total_seconds: float = NEGATIVE_TOTAL_SECONDS,
    clip_seconds: float = NEGATIVE_CLIP_SECONDS,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    overwrite: bool = False,
    input_func: InputFunction = input,
    recorder: Callable[[float], np.ndarray] | None = None,
) -> list[ManifestEntry]:
    count = max(
        1,
        int(
            ceil(
                total_seconds / clip_seconds,
            )
        ),
    )

    return _collect_samples(
        kind="negative",
        count=count,
        duration_seconds=clip_seconds,
        dataset_root=dataset_root,
        overwrite=overwrite,
        input_func=input_func,
        recorder=recorder,
    )


def record_resampled_clip(
    duration_seconds: float,
    audio_config: WakeWordAudioConfig | None = None,
    sounddevice_module: object | None = None,
    frame_processor: WakeWordFrameProcessor | None = None,
) -> np.ndarray:
    if duration_seconds < MIN_CLIP_SECONDS:
        raise WakeWordAudioError(
            "El clip es demasiado corto para el dataset."
        )

    config = audio_config or WakeWordAudioConfig()
    processor = frame_processor or WakeWordFrameProcessor(
        config=config,
    )
    sd = sounddevice_module or _import_sounddevice()
    target_samples = int(
        config.model_sample_rate
        * duration_seconds
    )
    native_blocks = int(
        ceil(
            duration_seconds
            * 1000
            / config.frame_ms
        )
    ) + 2
    output_chunks: list[np.ndarray] = []

    try:
        sd.check_input_settings(
            device=config.device_index,
            samplerate=config.native_sample_rate,
            channels=config.channels,
            dtype="int16",
        )

        with sd.InputStream(
            device=config.device_index,
            samplerate=config.native_sample_rate,
            channels=config.channels,
            dtype="int16",
            blocksize=config.native_block_samples,
        ) as stream:
            for _ in range(native_blocks):
                block, _ = stream.read(
                    config.native_block_samples,
                )
                output_chunks.extend(
                    processor.process_native_block(
                        block,
                    )
                )
                if _sample_count(output_chunks) >= target_samples:
                    break
    except Exception as error:
        raise WakeWordAudioError(
            f"No pude grabar muestra wake word: {error}"
        ) from error

    if not output_chunks:
        raise WakeWordAudioError(
            "La captura no produjo audio remuestreado."
        )

    samples = np.concatenate(
        output_chunks,
    )[
        :target_samples
    ]

    if samples.size < target_samples:
        raise WakeWordAudioError(
            "La captura produjo menos muestras de las esperadas."
        )

    return np.ascontiguousarray(
        samples,
        dtype=np.int16,
    )


def save_dataset_sample(
    samples: np.ndarray,
    output_path: Path,
    overwrite: bool = False,
) -> AudioMetrics:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"La muestra ya existe y no se sobrescribira: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    mono_samples = _mono_pcm16(
        samples,
    )
    metrics = calculate_metrics(
        mono_samples,
        sample_rate=settings.WAKEWORD_MODEL_SAMPLE_RATE,
    )

    if metrics.duration_seconds < MIN_CLIP_SECONDS:
        raise WakeWordAudioError(
            "El clip es demasiado corto para el dataset."
        )

    _write_wav_pcm16(
        output_path,
        mono_samples,
        sample_rate=settings.WAKEWORD_MODEL_SAMPLE_RATE,
    )

    return metrics


def calculate_metrics(
    samples: np.ndarray,
    sample_rate: int,
) -> AudioMetrics:
    mono_samples = _mono_pcm16(
        samples,
    )
    duration_seconds = (
        mono_samples.size
        / sample_rate
        if sample_rate
        else 0.0
    )

    if mono_samples.size == 0:
        peak = 0.0
        rms = 0.0
    else:
        float_samples = (
            mono_samples.astype(
                np.float32,
            )
            / 32768.0
        )
        peak = float(
            np.max(
                np.abs(
                    float_samples,
                )
            )
        )
        rms = float(
            sqrt(
                float(
                    np.mean(
                        float_samples
                        * float_samples,
                    )
                )
            )
        )

    return AudioMetrics(
        duration_seconds=round(
            duration_seconds,
            3,
        ),
        sample_rate=sample_rate,
        channels=1,
        peak_dbfs=_dbfs(
            peak,
        ),
        rms_dbfs=_dbfs(
            rms,
        ),
        peak=round(
            peak,
            6,
        ),
        rms=round(
            rms,
            6,
        ),
    )


def sample_path(
    kind: str,
    index: int,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> Path:
    if kind not in {
        "positive",
        "negative",
    }:
        raise ValueError(
            f"Tipo de muestra no soportado: {kind}"
        )

    return (
        dataset_root
        / kind
        / f"orion_{kind}_{index:03d}.wav"
    )


def append_manifest_entries(
    entries: Sequence[ManifestEntry],
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> None:
    manifest_path = dataset_root / MANIFEST_FILENAME
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    existing = _load_manifest(
        manifest_path,
    )
    existing.extend(
        asdict(entry)
        for entry in entries
    )
    manifest_path.write_text(
        json.dumps(
            existing,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(
    argv: Sequence[str] | None = None,
    input_func: InputFunction = input,
) -> int:
    parser = argparse.ArgumentParser(
        description="Recolector local de muestras wake word Orion."
    )
    subparsers = parser.add_subparsers(
        dest="kind",
        required=True,
    )

    positive = subparsers.add_parser(
        "positive",
        help="Recolecta muestras positivas diciendo Orion.",
    )
    positive.add_argument(
        "--count",
        type=int,
        default=POSITIVE_COUNT,
    )
    positive.add_argument(
        "--duration",
        type=float,
        default=NEGATIVE_CLIP_SECONDS,
    )
    positive.add_argument(
        "--overwrite",
        action="store_true",
    )

    negative = subparsers.add_parser(
        "negative",
        help="Recolecta clips negativos sin decir Orion.",
    )
    negative.add_argument(
        "--total-seconds",
        type=float,
        default=NEGATIVE_TOTAL_SECONDS,
    )
    negative.add_argument(
        "--clip-seconds",
        type=float,
        default=NEGATIVE_CLIP_SECONDS,
    )
    negative.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    args = parser.parse_args(
        argv,
    )

    try:
        if args.kind == "positive":
            entries = collect_positive_samples(
                count=args.count,
                duration_seconds=args.duration,
                dataset_root=args.dataset_root,
                overwrite=args.overwrite,
                input_func=input_func,
            )
        else:
            entries = collect_negative_samples(
                total_seconds=args.total_seconds,
                clip_seconds=args.clip_seconds,
                dataset_root=args.dataset_root,
                overwrite=args.overwrite,
                input_func=input_func,
            )
    except WakeWordError as error:
        print(f"ORION [DATASET] > Error: {error}")
        return 1
    except FileExistsError as error:
        print(f"ORION [DATASET] > Error: {error}")
        return 1

    print(
        "ORION [DATASET] > Muestras guardadas: "
        f"{len(entries)}"
    )
    print(
        "ORION [DATASET] > Dataset: "
        f"{args.dataset_root}"
    )
    return 0


def _collect_samples(
    kind: str,
    count: int,
    duration_seconds: float,
    dataset_root: Path,
    overwrite: bool,
    input_func: InputFunction,
    recorder: Callable[[float], np.ndarray] | None,
) -> list[ManifestEntry]:
    if count <= 0:
        raise WakeWordAudioError(
            "La cantidad de muestras debe ser mayor que cero."
        )

    (dataset_root / kind).mkdir(
        parents=True,
        exist_ok=True,
    )

    entries: list[ManifestEntry] = []
    active_recorder = recorder or record_resampled_clip

    for index in range(
        1,
        count + 1,
    ):
        instruction = _instruction_for(
            kind,
            index,
        )
        output_path = sample_path(
            kind,
            index,
            dataset_root,
        )
        relative_path = output_path.relative_to(
            dataset_root,
        )

        print(f"Muestra {index}/{count}")
        print("Presiona Enter")
        input_func("")
        print(instruction)
        if kind == "negative":
            print(NEGATIVE_INSTRUCTION)
        print(
            "Grabando "
            f"{duration_seconds:g} segundos..."
        )

        samples = active_recorder(
            duration_seconds,
        )
        metrics = save_dataset_sample(
            samples=samples,
            output_path=output_path,
            overwrite=overwrite,
        )

        if metrics.peak == 0.0:
            print(
                "ORION [DATASET] > Advertencia: "
                "la muestra parece silencio absoluto."
            )

        entries.append(
            ManifestEntry(
                tipo=kind,
                ruta_relativa=relative_path.as_posix(),
                duracion=metrics.duration_seconds,
                sample_rate=metrics.sample_rate,
                canales=metrics.channels,
                peak_dbfs=metrics.peak_dbfs,
                rms_dbfs=metrics.rms_dbfs,
                fecha=datetime.now(timezone.utc).isoformat(),
                instruccion=instruction,
            )
        )

    append_manifest_entries(
        entries,
        dataset_root=dataset_root,
    )

    return entries


def _instruction_for(
    kind: str,
    index: int,
) -> str:
    if kind == "positive":
        variation = POSITIVE_INSTRUCTIONS[
            (
                index
                - 1
            )
            % len(POSITIVE_INSTRUCTIONS)
        ]
        return f"Di Orion ({variation})."

    return "Graba habla o ambiente sin la palabra Orion."


def _write_wav_pcm16(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    with wave.open(
        str(path),
        "wb",
    ) as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(
            _mono_pcm16(
                samples,
            ).tobytes()
        )


def _mono_pcm16(
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
    )


def _load_manifest(
    manifest_path: Path,
) -> list[dict[str, object]]:
    if not manifest_path.exists():
        return []

    return json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )


def _sample_count(
    chunks: Sequence[np.ndarray],
) -> int:
    return sum(
        int(chunk.size)
        for chunk in chunks
    )


def _dbfs(
    level: float,
) -> float:
    if level <= 0.0:
        return float("-inf")

    return round(
        20
        * log10(
            level,
        ),
        2,
    )


def _import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as error:
        raise WakeWordAudioError(
            "Falta sounddevice para grabar muestras."
        ) from error

    return sd


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
