from __future__ import annotations

import sys
from types import SimpleNamespace

from orion.wakeword import diagnostic
from orion.wakeword.models import WakeWordEvent, WakeWordDebugStats


def test_diagnostic_reports_detection_without_pipeline_modules(
    monkeypatch,
    capsys,
) -> None:
    for module_name in (
        "faster_whisper",
        "openwakeword",
        "orion.llm.ollama_client",
        "orion.execution.service",
    ):
        monkeypatch.delitem(
            sys.modules,
            module_name,
            raising=False,
        )

    monkeypatch.setattr(
        diagnostic,
        "describe_microphone",
        lambda audio_config: ("Realtek", 48_000),
    )
    monkeypatch.setattr(
        diagnostic,
        "WakeWordService",
        _FakeDiagnosticService,
    )

    code = diagnostic.main(
        ["--duration", "1"]
    )

    output = capsys.readouterr().out

    assert code == 0
    assert "Microfono: Realtek, 48000 Hz" in output
    assert "Modelo: hey jarvis" in output
    assert "Detectado: hey jarvis" in output
    assert "Score: 0.82" in output
    assert "faster_whisper" not in sys.modules
    assert "openwakeword" not in sys.modules
    assert "orion.llm.ollama_client" not in sys.modules
    assert "orion.execution.service" not in sys.modules


def test_diagnostic_debug_prints_aggregate_metrics(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        diagnostic,
        "describe_microphone",
        lambda audio_config: ("Realtek", 48_000),
    )
    monkeypatch.setattr(
        diagnostic,
        "WakeWordService",
        _FakeDiagnosticService,
    )

    code = diagnostic.main(
        ["--duration", "1", "--debug"]
    )

    output = capsys.readouterr().out

    assert code == 0
    assert "sample rate nativo: 48000" in output
    assert "sample rate del modelo: 16000" in output
    assert "bloques de inferencia: 3" in output
    assert "score maximo: 0.82" in output
    assert "overflows: 1" in output


def test_prepare_models_is_explicit(
    monkeypatch,
    capsys,
) -> None:
    calls = {
        "download": None,
    }

    def prepare(
        download: bool,
    ) -> str:
        calls["download"] = download
        return "C:/models/openwakeword"

    monkeypatch.setattr(
        diagnostic,
        "prepare_wakeword_resources",
        prepare,
    )

    code = diagnostic.main(
        ["--prepare-models"]
    )

    output = capsys.readouterr().out

    assert code == 0
    assert calls["download"] is True
    assert "Recursos en: C:/models/openwakeword" in output


class _FakeDiagnosticService:
    model_name = "hey jarvis"

    def __init__(
        self,
        **kwargs: object,
    ) -> None:
        self.kwargs = kwargs

    def listen(
        self,
        duration_seconds: float,
    ) -> WakeWordEvent:
        return WakeWordEvent(
            wake_word="hey jarvis",
            score=0.82,
            detected_at=1.0,
        )

    def debug_stats(
        self,
        total_ms: float,
    ) -> WakeWordDebugStats:
        return WakeWordDebugStats(
            native_sample_rate=48_000,
            model_sample_rate=16_000,
            native_frames_received=11_520,
            model_frames_produced=3_840,
            inference_blocks=3,
            max_score=0.82,
            average_score=0.25,
            overflows=1,
            status_count=0,
            total_ms=total_ms,
        )
