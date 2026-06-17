import pytest

from orion import main as orion_main
from orion.llm.models import (
    IntentInterpretation,
    IntentType,
)
from orion.models import ToolResult
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
    capsys,
) -> None:
    pipeline = _FakePipeline(
        _allow_result(
            execution_enabled=False,
        )
    )

    code = orion_main.main(
        [],
        pipeline_factory=lambda: pipeline,
    )

    output = capsys.readouterr().out

    assert code == orion_main.EXIT_OK
    assert pipeline.calls == [False]
    assert "Modo: simulacion" in output
    assert "Ejecucion: deshabilitada" in output


def test_main_with_execute_enables_execution(
    capsys,
) -> None:
    pipeline = _FakePipeline(
        _allow_result(
            execution_enabled=True,
            execution_attempted=True,
            tool_result=ToolResult(
                success=True,
                tool_name="open_application",
                message="Abri Calculadora.",
            ),
        )
    )

    code = orion_main.main(
        ["--execute"],
        pipeline_factory=lambda: pipeline,
    )

    output = capsys.readouterr().out

    assert code == orion_main.EXIT_OK
    assert pipeline.calls == [True]
    assert "Modo: ejecucion real" in output
    assert "Ejecucion: intentada" in output
    assert "Resultado: Abri Calculadora." in output


def test_conversation_prints_reply_and_does_not_execute(
    capsys,
) -> None:
    pipeline = _FakePipeline(
        _conversation_result()
    )

    code = orion_main.main(
        ["--execute"],
        pipeline_factory=lambda: pipeline,
    )

    output = capsys.readouterr().out

    assert code == orion_main.EXIT_OK
    assert pipeline.calls == [True]
    assert "Respuesta: Hola, en que puedo ayudarte?" in output
    assert "Ejecucion: no requerida" in output


def test_unknown_prints_clarification_and_does_not_execute(
    capsys,
) -> None:
    pipeline = _FakePipeline(
        _unknown_result()
    )

    code = orion_main.main(
        ["--execute"],
        pipeline_factory=lambda: pipeline,
    )

    output = capsys.readouterr().out

    assert code == orion_main.EXIT_OK
    assert pipeline.calls == [True]
    assert "Aclaracion: No entendi la solicitud." in output
    assert "Ejecucion: no requerida" in output


def test_allow_in_simulation_does_not_execute(
    capsys,
) -> None:
    pipeline = _FakePipeline(
        _allow_result(
            execution_enabled=False,
        )
    )

    code = orion_main.main(
        [],
        pipeline_factory=lambda: pipeline,
    )

    output = capsys.readouterr().out

    assert code == orion_main.EXIT_OK
    assert pipeline.calls == [False]
    assert "Politica: ALLOW" in output
    assert "Ejecucion: deshabilitada" in output
    assert (
        "Resultado: La accion fue permitida, pero la ejecucion real "
        "esta deshabilitada."
    ) in output


def test_allow_with_execute_executes_once(
    capsys,
) -> None:
    pipeline = _FakePipeline(
        _allow_result(
            execution_enabled=True,
            execution_attempted=True,
            tool_result=ToolResult(
                success=True,
                tool_name="open_application",
                message="Abri Calculadora.",
            ),
        )
    )

    code = orion_main.main(
        ["--execute"],
        pipeline_factory=lambda: pipeline,
    )

    assert code == orion_main.EXIT_OK
    assert pipeline.calls == [True]
    assert "Resultado: Abri Calculadora." in capsys.readouterr().out


def test_block_never_executes_and_returns_success(
    capsys,
) -> None:
    pipeline = _FakePipeline(
        _block_result(
            execution_enabled=True,
        )
    )

    code = orion_main.main(
        ["--execute"],
        pipeline_factory=lambda: pipeline,
    )

    output = capsys.readouterr().out

    assert code == orion_main.EXIT_OK
    assert pipeline.calls == [True]
    assert "Politica: BLOCK" in output
    assert "Ejecucion: bloqueada" in output
    assert "Resultado: Solicitud bloqueada por seguridad." in output


def test_capture_error_returns_exit_code_2() -> None:
    pipeline = _FakePipeline(
        _error_result(
            message="No pude acceder al microfono.",
            error_stage="capture_transcription",
        )
    )

    code = orion_main.main(
        [],
        pipeline_factory=lambda: pipeline,
    )

    assert code == orion_main.EXIT_CAPTURE_OR_TRANSCRIPTION_ERROR


def test_ollama_error_returns_exit_code_3() -> None:
    pipeline = _FakePipeline(
        _error_result(
            message="Fallo Ollama: timeout",
            transcript="Abre la calculadora",
            error_stage="llm",
        )
    )

    code = orion_main.main(
        [],
        pipeline_factory=lambda: pipeline,
    )

    assert code == orion_main.EXIT_OLLAMA_ERROR


def test_configuration_error_returns_exit_code_4() -> None:
    pipeline = _FakePipeline(
        _error_result(
            message="Fallo Ollama: OLLAMA_BASE_URL invalida",
            transcript="Abre la calculadora",
            error_stage="configuration",
        )
    )

    code = orion_main.main(
        [],
        pipeline_factory=lambda: pipeline,
    )

    assert code == orion_main.EXIT_CONFIGURATION_ERROR


def test_pipeline_factory_configuration_error_returns_exit_code_4(
    capsys,
) -> None:
    code = orion_main.main(
        [],
        pipeline_factory=lambda: (_raise_value_error()),
    )

    assert code == orion_main.EXIT_CONFIGURATION_ERROR
    assert "Error de configuracion" in capsys.readouterr().out


def test_help_works_without_building_pipeline(
    capsys,
) -> None:
    called = {
        "value": False,
    }

    def factory() -> _FakePipeline:
        called["value"] = True
        return _FakePipeline(
            _conversation_result()
        )

    with pytest.raises(SystemExit) as error:
        orion_main.main(
            ["--help"],
            pipeline_factory=factory,
        )

    output = capsys.readouterr().out

    assert error.value.code == 0
    assert called["value"] is False
    assert "--execute" in output
    assert "Sin este" in output
    assert "solo simula" in output


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
    execution_attempted: bool = False,
    tool_result: ToolResult | None = None,
) -> VoicePipelineResult:
    interpretation = IntentInterpretation(
        original_text="Abre la calculadora",
        normalized_text="abre la calculadora",
        intent=IntentType.OPEN_APPLICATION,
        application_name="calculadora",
    )
    action_request = _action_request(
        interpretation,
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
        transcript=interpretation.original_text,
        interpretation=interpretation,
        action_request=action_request,
        policy_decision=policy_decision,
        tool_result=tool_result,
        execution_enabled=execution_enabled,
        timings_ms=_timings(),
        success=(
            tool_result.success
            if tool_result is not None
            else True
        ),
        message=(
            tool_result.message
            if tool_result is not None
            else "Simulacion: ejecucion real deshabilitada."
        ),
        execution_attempted=execution_attempted,
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
    action_request = _action_request(
        interpretation,
    )
    policy_decision = PolicyDecision(
        decision=PolicyDecisionType.BLOCK,
        risk_level=RiskLevel.LOW,
        reason="application_name esta bloqueada por politica",
        tool_name="open_application",
        arguments=action_request.arguments,
        evaluation_ms=1.0,
    )

    return VoicePipelineResult(
        transcript=interpretation.original_text,
        interpretation=interpretation,
        action_request=action_request,
        policy_decision=policy_decision,
        tool_result=None,
        execution_enabled=execution_enabled,
        timings_ms=_timings(),
        success=True,
        message="Bloqueado por politica.",
        execution_attempted=False,
    )


def _conversation_result() -> VoicePipelineResult:
    interpretation = IntentInterpretation(
        original_text="Hola Orion.",
        normalized_text="hola orion",
        intent=IntentType.CONVERSATION,
        assistant_reply="Hola, en que puedo ayudarte?",
    )

    return VoicePipelineResult(
        transcript=interpretation.original_text,
        interpretation=interpretation,
        action_request=None,
        policy_decision=None,
        tool_result=None,
        execution_enabled=True,
        timings_ms=_timings(),
        success=True,
        message=interpretation.assistant_reply or "",
    )


def _unknown_result() -> VoicePipelineResult:
    interpretation = IntentInterpretation(
        original_text="Prondo.",
        normalized_text="prondo",
        intent=IntentType.UNKNOWN,
        needs_clarification=True,
        clarification_question="No entendi la solicitud.",
    )

    return VoicePipelineResult(
        transcript=interpretation.original_text,
        interpretation=interpretation,
        action_request=None,
        policy_decision=None,
        tool_result=None,
        execution_enabled=True,
        timings_ms=_timings(),
        success=True,
        message=interpretation.clarification_question or "",
    )


def _error_result(
    message: str,
    error_stage: str,
    transcript: str = "",
) -> VoicePipelineResult:
    return VoicePipelineResult(
        transcript=transcript,
        interpretation=None,
        action_request=None,
        policy_decision=None,
        tool_result=None,
        execution_enabled=False,
        timings_ms=_timings(),
        success=False,
        message=message,
        execution_attempted=False,
        error_stage=error_stage,
    )


def _action_request(
    interpretation: IntentInterpretation,
) -> ActionRequest:
    return ActionRequest(
        tool_name="open_application",
        arguments={
            "application_name": interpretation.application_name or "",
        },
        original_text=interpretation.original_text,
        source=LOCAL_VOICE_SOURCE,
    )


def _timings() -> dict[str, float]:
    return {
        "capture": 5000.0,
        "transcription": 120.0,
        "llm": 40.0,
        "routing": 1.0,
        "policy": 1.0,
        "execution": 0.0,
        "total": 5162.0,
    }


def _raise_value_error() -> None:
    raise ValueError(
        "configuracion invalida"
    )
