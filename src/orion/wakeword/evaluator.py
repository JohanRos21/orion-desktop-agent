from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

from orion.config import settings
from orion.wakeword.exceptions import (
    UnsupportedWakeWordModelError,
    WakeWordDependencyError,
    WakeWordModelError,
)
from orion.wakeword.sample_collector import (
    DEFAULT_DATASET_ROOT,
    MANIFEST_FILENAME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ClipScorer = Callable[[Path], float]


@dataclass(frozen=True, slots=True)
class GroupScores:
    count: int
    activated: int
    rejected: int
    min_score: float
    max_score: float
    average_score: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    positives_detected: int
    positives_rejected: int
    detection_rate: float
    negatives_false_activated: int
    false_positive_rate: float
    positive_scores: GroupScores
    negative_scores: GroupScores


def evaluate_dataset(
    model_path: Path,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    threshold: float = settings.WAKEWORD_THRESHOLD,
    allowed_roots: Sequence[Path] | None = None,
    clip_scorer: ClipScorer | None = None,
) -> EvaluationResult:
    validated_model_path = validate_model_path(
        model_path,
        allowed_roots=allowed_roots,
    )
    entries = load_manifest_entries(
        dataset_root,
    )
    scorer = clip_scorer or _OpenWakeWordClipScorer(
        validated_model_path,
    )

    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for entry in entries:
        clip_path = dataset_root / str(
            entry["ruta_relativa"],
        )
        score = scorer(
            clip_path,
        )

        if entry["tipo"] == "positive":
            positive_scores.append(
                score,
            )
        elif entry["tipo"] == "negative":
            negative_scores.append(
                score,
            )

    positive_group = _group_scores(
        positive_scores,
        threshold,
    )
    negative_group = _group_scores(
        negative_scores,
        threshold,
    )

    positives_detected = positive_group.activated
    positives_rejected = positive_group.rejected
    negatives_false_activated = negative_group.activated

    return EvaluationResult(
        positives_detected=positives_detected,
        positives_rejected=positives_rejected,
        detection_rate=_rate(
            positives_detected,
            positive_group.count,
        ),
        negatives_false_activated=negatives_false_activated,
        false_positive_rate=_rate(
            negatives_false_activated,
            negative_group.count,
        ),
        positive_scores=positive_group,
        negative_scores=negative_group,
    )


def validate_model_path(
    model_path: Path,
    allowed_roots: Sequence[Path] | None = None,
) -> Path:
    path = model_path.resolve()

    if not path.exists():
        raise WakeWordModelError(
            f"No existe el modelo: {path}"
        )

    suffix = path.suffix.casefold()
    if suffix == ".tflite":
        raise UnsupportedWakeWordModelError(
            "El evaluador solo acepta modelos ONNX."
        )

    if suffix != ".onnx":
        raise WakeWordModelError(
            "El modelo debe terminar en .onnx."
        )

    roots = [
        root.resolve()
        for root in (
            allowed_roots
            or default_allowed_model_roots()
        )
    ]

    if not any(
        path.is_relative_to(
            root,
        )
        for root in roots
    ):
        raise WakeWordModelError(
            "La ruta del modelo esta fuera de las rutas locales permitidas."
        )

    return path


def default_allowed_model_roots() -> list[Path]:
    roots = [
        PROJECT_ROOT
        / "local_models"
        / "wakeword",
    ]

    try:
        import openwakeword
    except ImportError:
        return roots

    roots.append(
        Path(
            openwakeword.__file__,
        ).resolve().parent
        / "resources"
        / "models"
    )

    return roots


def load_manifest_entries(
    dataset_root: Path,
) -> list[dict[str, object]]:
    manifest_path = dataset_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        return []

    return json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )


class _OpenWakeWordClipScorer:
    def __init__(
        self,
        model_path: Path,
    ) -> None:
        try:
            from openwakeword.model import Model
        except ImportError as error:
            raise WakeWordDependencyError(
                "Falta openwakeword para evaluar modelos."
            ) from error

        self.model_name = model_path.stem
        self.model = Model(
            wakeword_models=[
                str(model_path),
            ],
            inference_framework="onnx",
            vad_threshold=0,
        )

    def __call__(
        self,
        clip_path: Path,
    ) -> float:
        samples = read_wav_pcm16(
            clip_path,
        )
        max_score = 0.0
        frame_samples = int(
            settings.WAKEWORD_MODEL_SAMPLE_RATE
            * settings.WAKEWORD_FRAME_MS
            / 1000
        )

        for index in range(
            0,
            samples.size,
            frame_samples,
        ):
            frame = samples[
                index : index + frame_samples
            ]
            if frame.size < frame_samples:
                break

            predictions = self.model.predict(
                frame,
            )
            score = max(
                (
                    float(value)
                    for value in predictions.values()
                ),
                default=0.0,
            )
            max_score = max(
                max_score,
                score,
            )

        return max_score


def read_wav_pcm16(
    path: Path,
) -> np.ndarray:
    with wave.open(
        str(path),
        "rb",
    ) as wav_file:
        if wav_file.getframerate() != settings.WAKEWORD_MODEL_SAMPLE_RATE:
            raise ValueError(
                "El WAV debe estar a 16000 Hz."
            )
        if wav_file.getnchannels() != 1:
            raise ValueError(
                "El WAV debe ser mono."
            )
        if wav_file.getsampwidth() != 2:
            raise ValueError(
                "El WAV debe ser PCM16."
            )
        audio_bytes = wav_file.readframes(
            wav_file.getnframes(),
        )

    return np.frombuffer(
        audio_bytes,
        dtype=np.int16,
    ).copy()


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Evalua un modelo wake word ONNX sobre el dataset local."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.WAKEWORD_THRESHOLD,
    )
    args = parser.parse_args(
        argv,
    )

    try:
        result = evaluate_dataset(
            model_path=args.model_path,
            dataset_root=args.dataset_root,
            threshold=args.threshold,
        )
    except Exception as error:
        print(f"ORION [EVALUADOR] > Error: {error}")
        return 1

    print_evaluation_result(
        result,
    )
    return 0


def print_evaluation_result(
    result: EvaluationResult,
) -> None:
    print(f"positivos detectados: {result.positives_detected}")
    print(f"positivos rechazados: {result.positives_rejected}")
    print(f"tasa de deteccion: {result.detection_rate:.2%}")
    print(
        "negativos activados falsamente: "
        f"{result.negatives_false_activated}"
    )
    print(
        "tasa de falsos positivos: "
        f"{result.false_positive_rate:.2%}"
    )
    _print_group(
        "positivos",
        result.positive_scores,
    )
    _print_group(
        "negativos",
        result.negative_scores,
    )


def _print_group(
    label: str,
    scores: GroupScores,
) -> None:
    print(
        f"{label} score min/max/promedio: "
        f"{scores.min_score:.4f} / "
        f"{scores.max_score:.4f} / "
        f"{scores.average_score:.4f}"
    )


def _group_scores(
    scores: Sequence[float],
    threshold: float,
) -> GroupScores:
    activated = sum(
        1
        for score in scores
        if score >= threshold
    )
    count = len(
        scores,
    )
    rejected = count - activated

    return GroupScores(
        count=count,
        activated=activated,
        rejected=rejected,
        min_score=round(
            min(
                scores,
                default=0.0,
            ),
            6,
        ),
        max_score=round(
            max(
                scores,
                default=0.0,
            ),
            6,
        ),
        average_score=round(
            (
                sum(scores)
                / count
            )
            if count
            else 0.0,
            6,
        ),
    )


def _rate(
    value: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return value / total


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
