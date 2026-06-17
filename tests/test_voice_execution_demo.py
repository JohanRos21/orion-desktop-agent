from orion.llm.models import (
    IntentInterpretation,
    IntentType,
)
from orion.orchestration import voice_execution_demo
from orion.orchestration.voice_pipeline import (
    LOCAL_VOICE_SOURCE,
    VoicePipelineResult,
)
from orion.policy.models import (
    PolicyDecision,
    PolicyDecisionType,
)
from orion.tools.contracts import (
    ActionRequest,
    RiskLevel,
)


def test_main_without_execute_uses_simulation(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def fake_run_demo(
        execution_enabled: bool,
    ) -> VoicePipelineResult:
        calls.append(
            execution_enabled,
        )
        return _allow_result(
            execution_enabled=execution_enabled,
        )

    monkeypatch.setattr(
        voice_execution_demo,
        "run_demo",
        fake_run_demo,
    )

    voice_execution_demo.main([])

    assert calls == [False]


def test_main_with_execute_explicitly_enables_execution(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def fake_run_demo(
        execution_enabled: bool,
    ) -> VoicePipelineResult:
        calls.append(
            execution_enabled,
        )
        return _allow_result(
            execution_enabled=execution_enabled,
        )

    monkeypatch.setattr(
        voice_execution_demo,
        "run_demo",
        fake_run_demo,
    )

    voice_execution_demo.main(["--execute"])

    assert calls == [True]


def test_demo_without_execute_does_not_execute(
    capsys,
) -> None:
    pipeline = _FakePipeline(
        _allow_result(
            execution_enabled=False,
        )
    )

    result = voice_execution_demo.run_demo(
        execution_enabled=False,
        pipeline=pipeline,
    )

    captured = capsys.readouterr()

    assert pipeline.calls == [False]
    assert result.execution_attempted is False
    assert "Ejecucion habilitada" in captured.out
    assert "False" in captured.out
    assert "ALLOW" in captured.out


def test_demo_with_execute_respects_policy_block(
    capsys,
) -> None:
    pipeline = _FakePipeline(
        _block_result(
            execution_enabled=True,
        )
    )

    result = voice_execution_demo.run_demo(
        execution_enabled=True,
        pipeline=pipeline,
    )

    captured = capsys.readouterr()

    assert pipeline.calls == [True]
    assert result.execution_enabled is True
    assert result.execution_attempted is False
    assert result.policy_decision is not None
    assert result.policy_decision.decision is PolicyDecisionType.BLOCK
    assert "BLOCK" in captured.out
    assert "Bloqueado por politica" in captured.out


class _FakePipeline:
    def __init__(
        self,
        result: VoicePipelineResult,
    ) -> None:
        self.result = result
        self.calls: list[bool] = []

    def run(
        self,
        execution_enabled: bool,
    ) -> VoicePipelineResult:
        self.calls.append(
            execution_enabled,
        )
        return self.result


def _allow_result(
    execution_enabled: bool,
) -> VoicePipelineResult:
    interpretation = IntentInterpretation(
        original_text="Abre la calculadora",
        normalized_text="abre la calculadora",
        intent=IntentType.OPEN_APPLICATION,
        application_name="calculadora",
    )
    action_request = ActionRequest(
        tool_name="open_application",
        arguments={
            "application_name": "calculadora",
        },
        original_text="Abre la calculadora",
        source=LOCAL_VOICE_SOURCE,
    )
    policy_decision = PolicyDecision(
        decision=PolicyDecisionType.ALLOW,
        risk_level=RiskLevel.LOW,
        reason="permitido",
        tool_name="open_application",
        arguments=action_request.arguments,
        evaluation_ms=1.0,
    )

    return VoicePipelineResult(
        transcript="Abre la calculadora",
        interpretation=interpretation,
        action_request=action_request,
        policy_decision=policy_decision,
        tool_result=None,
        execution_enabled=execution_enabled,
        timings_ms=_timings(),
        success=True,
        message="Simulacion: ejecucion real deshabilitada.",
        execution_attempted=False,
    )


def _block_result(
    execution_enabled: bool,
) -> VoicePipelineResult:
    interpretation = IntentInterpretation(
        original_text="Abre PowerShell",
        normalized_text="abre powershell",
        intent=IntentType.OPEN_APPLICATION,
        application_name="powershell",
    )
    action_request = ActionRequest(
        tool_name="open_application",
        arguments={
            "application_name": "powershell",
        },
        original_text="Abre PowerShell",
        source=LOCAL_VOICE_SOURCE,
    )
    policy_decision = PolicyDecision(
        decision=PolicyDecisionType.BLOCK,
        risk_level=RiskLevel.LOW,
        reason="application_name esta bloqueada",
        tool_name="open_application",
        arguments=action_request.arguments,
        evaluation_ms=1.0,
    )

    return VoicePipelineResult(
        transcript="Abre PowerShell",
        interpretation=interpretation,
        action_request=action_request,
        policy_decision=policy_decision,
        tool_result=None,
        execution_enabled=execution_enabled,
        timings_ms=_timings(),
        success=True,
        message="Bloqueado por politica: application_name esta bloqueada",
        execution_attempted=False,
    )


def _timings() -> dict[str, float]:
    return {
        "capture": 5000.0,
        "transcription": 100.0,
        "llm": 30.0,
        "routing": 1.0,
        "policy": 1.0,
        "execution": 0.0,
        "total": 5132.0,
    }
