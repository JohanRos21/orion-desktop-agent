from __future__ import annotations

import json
import sys
import wave

import numpy as np
import pytest

from orion.wakeword import sample_collector
from orion.wakeword.exceptions import WakeWordAudioError


def test_creates_positive_folders_numbered_samples_and_manifest(
    tmp_path,
) -> None:
    samples = np.full(
        32_000,
        1000,
        dtype=np.int16,
    )

    entries = sample_collector.collect_positive_samples(
        count=2,
        duration_seconds=2.0,
        dataset_root=tmp_path,
        input_func=lambda prompt: "",
        recorder=lambda duration: samples,
    )

    assert (tmp_path / "positive").exists()
    assert (tmp_path / "positive" / "orion_positive_001.wav").exists()
    assert (tmp_path / "positive" / "orion_positive_002.wav").exists()
    assert len(entries) == 2

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(
            encoding="utf-8",
        )
    )

    assert manifest[0]["tipo"] == "positive"
    assert manifest[0]["ruta_relativa"] == "positive/orion_positive_001.wav"
    assert manifest[0]["sample_rate"] == 16_000
    assert manifest[0]["canales"] == 1
    assert "fecha" in manifest[0]
    assert "Di Orion" in manifest[0]["instruccion"]


def test_negative_collection_uses_short_clips_and_warning(
    tmp_path,
    capsys,
) -> None:
    samples = np.full(
        32_000,
        500,
        dtype=np.int16,
    )

    entries = sample_collector.collect_negative_samples(
        total_seconds=4.0,
        clip_seconds=2.0,
        dataset_root=tmp_path,
        input_func=lambda prompt: "",
        recorder=lambda duration: samples,
    )

    output = capsys.readouterr().out

    assert len(entries) == 2
    assert "No pronuncies Orion" in output
    assert (tmp_path / "negative" / "orion_negative_001.wav").exists()
    assert (tmp_path / "negative" / "orion_negative_002.wav").exists()


def test_saved_wav_is_mono_pcm16_16khz(
    tmp_path,
) -> None:
    samples = np.array(
        [1, -2, 300, -400],
        dtype=np.int16,
    )
    output_path = tmp_path / "sample.wav"

    metrics = sample_collector.save_dataset_sample(
        np.tile(
            samples,
            4000,
        ),
        output_path,
    )

    with wave.open(
        str(output_path),
        "rb",
    ) as wav_file:
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2

    assert metrics.sample_rate == 16_000
    assert metrics.channels == 1


def test_does_not_overwrite_without_confirmation(
    tmp_path,
) -> None:
    output_path = tmp_path / "sample.wav"
    samples = np.ones(
        16_000,
        dtype=np.int16,
    )

    sample_collector.save_dataset_sample(
        samples,
        output_path,
    )

    with pytest.raises(FileExistsError):
        sample_collector.save_dataset_sample(
            samples,
            output_path,
        )


def test_too_short_clip_is_rejected(
    tmp_path,
) -> None:
    with pytest.raises(WakeWordAudioError):
        sample_collector.save_dataset_sample(
            np.zeros(
                100,
                dtype=np.int16,
            ),
            tmp_path / "short.wav",
        )


def test_silence_absolute_is_warned(
    tmp_path,
    capsys,
) -> None:
    sample_collector.collect_positive_samples(
        count=1,
        dataset_root=tmp_path,
        input_func=lambda prompt: "",
        recorder=lambda duration: np.zeros(
            32_000,
            dtype=np.int16,
        ),
    )

    output = capsys.readouterr().out

    assert "silencio absoluto" in output


def test_peak_and_rms_are_calculated() -> None:
    metrics = sample_collector.calculate_metrics(
        np.array(
            [0, 16384, -16384, 0],
            dtype=np.int16,
        ),
        sample_rate=16_000,
    )

    assert metrics.peak == 0.5
    assert metrics.peak_dbfs == pytest.approx(
        -6.02,
    )
    assert metrics.rms_dbfs == pytest.approx(
        -9.03,
    )


def test_cli_uses_local_dataset_only(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sample_collector,
        "record_resampled_clip",
        lambda duration: np.ones(
            32_000,
            dtype=np.int16,
        ),
    )

    code = sample_collector.main(
        [
            "--dataset-root",
            str(tmp_path),
            "positive",
            "--count",
            "1",
        ],
        input_func=lambda prompt: "",
    )

    assert code == 0
    assert tmp_path.exists()


def test_collector_does_not_import_pipeline_modules(
    monkeypatch,
    tmp_path,
) -> None:
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

    sample_collector.collect_positive_samples(
        count=1,
        dataset_root=tmp_path,
        input_func=lambda prompt: "",
        recorder=lambda duration: np.ones(
            32_000,
            dtype=np.int16,
        ),
    )

    assert "faster_whisper" not in sys.modules
    assert "orion.llm.ollama_client" not in sys.modules
    assert "orion.execution.service" not in sys.modules
