from __future__ import annotations

import json
import sys
import wave

import numpy as np
import pytest

from orion.wakeword import evaluator
from orion.wakeword.exceptions import (
    UnsupportedWakeWordModelError,
    WakeWordModelError,
)


def test_evaluates_positives_and_negatives(
    tmp_path,
) -> None:
    dataset_root = _dataset(
        tmp_path,
    )
    model_path = tmp_path / "models" / "orion.onnx"
    model_path.parent.mkdir()
    model_path.write_bytes(
        b"onnx",
    )
    scores = {
        "positive/orion_positive_001.wav": 0.80,
        "positive/orion_positive_002.wav": 0.40,
        "negative/orion_negative_001.wav": 0.20,
        "negative/orion_negative_002.wav": 0.70,
    }

    result = evaluator.evaluate_dataset(
        model_path=model_path,
        dataset_root=dataset_root,
        threshold=0.50,
        allowed_roots=[tmp_path / "models"],
        clip_scorer=lambda path: scores[
            path.relative_to(
                dataset_root,
            ).as_posix()
        ],
    )

    assert result.positives_detected == 1
    assert result.positives_rejected == 1
    assert result.detection_rate == 0.5
    assert result.negatives_false_activated == 1
    assert result.false_positive_rate == 0.5
    assert result.positive_scores.max_score == 0.8
    assert result.negative_scores.average_score == 0.45


def test_threshold_is_configurable(
    tmp_path,
) -> None:
    dataset_root = _dataset(
        tmp_path,
    )
    model_path = tmp_path / "models" / "orion.onnx"
    model_path.parent.mkdir()
    model_path.write_bytes(
        b"onnx",
    )

    result = evaluator.evaluate_dataset(
        model_path=model_path,
        dataset_root=dataset_root,
        threshold=0.90,
        allowed_roots=[tmp_path / "models"],
        clip_scorer=lambda path: 0.80,
    )

    assert result.positives_detected == 0
    assert result.positives_rejected == 2
    assert result.negatives_false_activated == 0


def test_missing_model_is_rejected(
    tmp_path,
) -> None:
    with pytest.raises(WakeWordModelError):
        evaluator.validate_model_path(
            tmp_path / "missing.onnx",
            allowed_roots=[tmp_path],
        )


def test_non_onnx_model_is_rejected(
    tmp_path,
) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(
        b"bad",
    )

    with pytest.raises(WakeWordModelError):
        evaluator.validate_model_path(
            model_path,
            allowed_roots=[tmp_path],
        )


def test_tflite_model_is_rejected(
    tmp_path,
) -> None:
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(
        b"bad",
    )

    with pytest.raises(UnsupportedWakeWordModelError):
        evaluator.validate_model_path(
            model_path,
            allowed_roots=[tmp_path],
        )


def test_model_outside_allowed_root_is_rejected(
    tmp_path,
) -> None:
    model_path = tmp_path / "outside" / "model.onnx"
    model_path.parent.mkdir()
    model_path.write_bytes(
        b"onnx",
    )

    with pytest.raises(WakeWordModelError):
        evaluator.validate_model_path(
            model_path,
            allowed_roots=[tmp_path / "allowed"],
        )


def test_reads_wav_pcm16() -> None:
    with pytest.raises(FileNotFoundError):
        evaluator.read_wav_pcm16(
            __import__("pathlib").Path("missing.wav"),
        )


def test_prints_evaluation_result(capsys) -> None:
    result = evaluator.EvaluationResult(
        positives_detected=2,
        positives_rejected=1,
        detection_rate=2 / 3,
        negatives_false_activated=1,
        false_positive_rate=0.25,
        positive_scores=evaluator.GroupScores(
            count=3,
            activated=2,
            rejected=1,
            min_score=0.1,
            max_score=0.9,
            average_score=0.5,
        ),
        negative_scores=evaluator.GroupScores(
            count=4,
            activated=1,
            rejected=3,
            min_score=0.0,
            max_score=0.8,
            average_score=0.2,
        ),
    )

    evaluator.print_evaluation_result(
        result,
    )

    output = capsys.readouterr().out

    assert "positivos detectados: 2" in output
    assert "tasa de deteccion: 66.67%" in output
    assert "tasa de falsos positivos: 25.00%" in output


def test_evaluator_does_not_import_pipeline_modules(monkeypatch) -> None:
    for module_name in (
        "faster_whisper",
        "orion.llm.ollama_client",
        "orion.execution.service",
    ):
        monkeypatch.delitem(
            sys.modules,
            module_name,
            raising=False,
        )

    evaluator._group_scores(
        [0.1, 0.9],
        threshold=0.5,
    )

    assert "faster_whisper" not in sys.modules
    assert "orion.llm.ollama_client" not in sys.modules
    assert "orion.execution.service" not in sys.modules


def _dataset(
    tmp_path,
):
    dataset_root = tmp_path / "dataset"
    for kind in (
        "positive",
        "negative",
    ):
        (dataset_root / kind).mkdir(
            parents=True,
        )

    manifest = []
    for kind, count in (
        ("positive", 2),
        ("negative", 2),
    ):
        for index in range(
            1,
            count + 1,
        ):
            relative_path = f"{kind}/orion_{kind}_{index:03d}.wav"
            _write_wav(
                dataset_root / relative_path,
            )
            manifest.append(
                {
                    "tipo": kind,
                    "ruta_relativa": relative_path,
                    "duracion": 1.0,
                    "sample_rate": 16_000,
                    "canales": 1,
                    "peak_dbfs": -6.0,
                    "rms_dbfs": -12.0,
                    "fecha": "2026-06-18T00:00:00+00:00",
                    "instruccion": "test",
                }
            )

    (dataset_root / "manifest.json").write_text(
        json.dumps(
            manifest,
        ),
        encoding="utf-8",
    )

    return dataset_root


def _write_wav(
    path,
) -> None:
    samples = np.zeros(
        16_000,
        dtype=np.int16,
    )
    with wave.open(
        str(path),
        "wb",
    ) as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(
            samples.tobytes(),
        )
