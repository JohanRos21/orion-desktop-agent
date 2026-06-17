import pytest

from orion.llm.exceptions import OllamaConnectionError
from orion.llm.models import (
    IntentInterpretation,
    IntentParseResult,
    IntentType,
)
from orion.models import ToolResult
from orion.orchestration import voice_pipeline
from orion.orchestration.voice_pipeline import (
    AUDIT_LOG,
    LOCAL_VOICE_SOURCE,
    TIMING_KEYS,
    VoicePipeline,
)
from orion.policy.engine import PolicyEngine
from orion.policy.models import (
    PolicyDecision,
    PolicyDecisionType,
)
from orion.tools.contracts import RiskLevel
from orion.tools.default_registry import build_default_registry
from orion.voice import VoiceResult


@pytest.fixture(autouse=True)
def clear_audit_log() -> None:
    AUDIT_LOG.clear()


def test_conversation_does_not_execute() -> None:
    execution_service = _FakeExecutionService()
    result = _pipeline(
        interpretation=IntentInterpretation(
            original_text="Hola Orion",
            normalized_text="hola orion",
            intent=IntentType.CONVERSATION,
            assistant_reply="Hola.",
        ),
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.action_request is None
    assert result.policy_decision is None
    assert result.tool_result is None
    assert result.execution_attempted is False
    assert result.message == "Hola."
    assert execution_service.calls == []


def test_unknown_does_not_execute() -> None:
    execution_service = _FakeExecutionService()
    result = _pipeline(
        interpretation=IntentInterpretation(
            original_text="Haz eso",
            normalized_text="haz eso",
            intent=IntentType.UNKNOWN,
            needs_clarification=True,
            clarification_question="Que accion quieres realizar?",
        ),
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.action_request is None
    assert result.execution_attempted is False
    assert result.message == "Que accion quieres realizar?"
    assert execution_service.calls == []


def test_empty_transcription_does_not_interpret_or_execute() -> None:
    execution_service = _FakeExecutionService()
    interpreted = {
        "value": False,
    }

    def interpreter(text: str) -> IntentParseResult:
        interpreted["value"] = True
        raise AssertionError("no debe interpretar texto vacio")

    result = VoicePipeline(
        voice_listener=lambda: _voice_result(
            transcript="   ",
        ),
        intent_interpreter=interpreter,
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.success is False
    assert result.interpretation is None
    assert result.execution_attempted is False
    assert interpreted["value"] is False
    assert execution_service.calls == []


def test_open_application_allow_with_execution_disabled_does_not_execute() -> None:
    execution_service = _FakeExecutionService()
    result = _pipeline(
        interpretation=_open_application_interpretation(
            text="Abre la calculadora",
            application_name="calculadora",
        ),
        execution_service=execution_service,
    ).run(execution_enabled=False)

    assert result.policy_decision is not None
    assert result.policy_decision.decision is PolicyDecisionType.ALLOW
    assert result.execution_enabled is False
    assert result.execution_attempted is False
    assert result.tool_result is None
    assert execution_service.calls == []


def test_open_application_allow_with_execution_enabled_executes_once() -> None:
    execution_service = _FakeExecutionService()
    result = _pipeline(
        interpretation=_open_application_interpretation(
            text="Abre la calculadora",
            application_name="calculadora",
        ),
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.policy_decision is not None
    assert result.policy_decision.decision is PolicyDecisionType.ALLOW
    assert result.execution_attempted is True
    assert result.tool_result is not None
    assert result.tool_result.success is True
    assert execution_service.calls == ["open_application"]


def test_block_never_executes() -> None:
    execution_service = _FakeExecutionService()
    result = _pipeline(
        interpretation=_open_application_interpretation(
            text="Abre PowerShell",
            application_name="powershell",
        ),
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.policy_decision is not None
    assert result.policy_decision.decision is PolicyDecisionType.BLOCK
    assert result.execution_attempted is False
    assert execution_service.calls == []


def test_confirm_never_executes() -> None:
    execution_service = _FakeExecutionService()
    result = _pipeline(
        interpretation=_open_application_interpretation(
            text="Abre la calculadora",
            application_name="calculadora",
        ),
        policy_engine=_StaticPolicyEngine(
            PolicyDecision(
                decision=PolicyDecisionType.CONFIRM,
                risk_level=RiskLevel.MEDIUM,
                reason="requiere confirmacion",
                tool_name="open_application",
                arguments={
                    "application_name": "calculadora",
                },
                evaluation_ms=2.0,
            )
        ),
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.policy_decision is not None
    assert result.policy_decision.decision is PolicyDecisionType.CONFIRM
    assert result.execution_attempted is False
    assert result.tool_result is None
    assert execution_service.calls == []


def test_ollama_failure_records_llm_timing_and_never_executes(
    monkeypatch,
) -> None:
    execution_service = _FakeExecutionService()
    times = iter(
        [
            10.0,
            10.1,
            10.35,
            10.5,
        ]
    )
    monkeypatch.setattr(
        voice_pipeline,
        "perf_counter",
        lambda: next(times),
    )

    def interpreter(text: str) -> IntentParseResult:
        raise OllamaConnectionError("ollama apagado")

    result = VoicePipeline(
        voice_listener=lambda: _voice_result(
            transcript="Abre la calculadora",
        ),
        intent_interpreter=interpreter,
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.success is False
    assert result.interpretation is None
    assert result.timings_ms["llm"] == 250.0
    assert result.execution_attempted is False
    assert execution_service.calls == []


def test_capture_failure_never_executes() -> None:
    execution_service = _FakeExecutionService()

    def listener() -> VoiceResult:
        raise RuntimeError("microfono no disponible")

    result = VoicePipeline(
        voice_listener=listener,
        intent_interpreter=lambda text: pytest.fail(
            "no debe invocar Ollama"
        ),
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.success is False
    assert "microfono no disponible" in result.message
    assert result.execution_attempted is False
    assert execution_service.calls == []


def test_executor_failure_returns_controlled_result() -> None:
    execution_service = _FakeExecutionService(
        error=RuntimeError("executor roto"),
    )
    result = _pipeline(
        interpretation=_open_application_interpretation(
            text="Abre la calculadora",
            application_name="calculadora",
        ),
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.execution_attempted is True
    assert result.tool_result is not None
    assert result.tool_result.success is False
    assert "executor roto" in result.tool_result.message


def test_action_source_is_local_voice() -> None:
    result = _pipeline(
        interpretation=_open_application_interpretation(
            text="Abre la calculadora",
            application_name="calculadora",
        ),
    ).run(execution_enabled=False)

    assert result.action_request is not None
    assert result.action_request.source == LOCAL_VOICE_SOURCE


def test_timings_are_present() -> None:
    result = _pipeline(
        interpretation=_open_application_interpretation(
            text="Abre la calculadora",
            application_name="calculadora",
        ),
    ).run(execution_enabled=False)

    assert set(TIMING_KEYS).issubset(
        result.timings_ms
    )
    assert result.timings_ms["capture"] == 5000.0
    assert result.timings_ms["transcription"] == 120.0
    assert result.timings_ms["llm"] == 30.0


def test_audit_record_is_created() -> None:
    result = _pipeline(
        interpretation=_open_application_interpretation(
            text="Abre la calculadora",
            application_name="calculadora",
        ),
    ).run(execution_enabled=False)

    assert len(AUDIT_LOG) == 1
    assert result.audit_record is AUDIT_LOG[0]
    assert result.audit_record is not None
    assert result.audit_record.transcript == "Abre la calculadora"
    assert result.audit_record.normalized_text == "abre la calculadora"
    assert result.audit_record.intent == "open_application"
    assert result.audit_record.tool_name == "open_application"
    assert result.audit_record.arguments == {
        "application_name": "calculadora",
    }
    assert result.audit_record.risk == "low"
    assert result.audit_record.decision == "allow"
    assert result.audit_record.execution_enabled is False
    assert result.audit_record.execution_attempted is False
    assert result.audit_record.execution_success is False
    assert result.audit_record.capture_ms == 5000.0
    assert result.audit_record.transcription_ms == 120.0
    assert result.audit_record.llm_ms == 30.0


def test_dangerous_original_text_is_blocked_without_partial_execution() -> None:
    execution_service = _FakeExecutionService()
    result = _pipeline(
        interpretation=_open_application_interpretation(
            text="Abre calculadora && borra archivos",
            application_name="calculadora",
        ),
        execution_service=execution_service,
    ).run(execution_enabled=True)

    assert result.policy_decision is not None
    assert result.policy_decision.decision is PolicyDecisionType.BLOCK
    assert result.execution_attempted is False
    assert execution_service.calls == []


def _pipeline(
    interpretation: IntentInterpretation,
    policy_engine: object | None = None,
    execution_service: object | None = None,
) -> VoicePipeline:
    return VoicePipeline(
        voice_listener=lambda: _voice_result(
            transcript=interpretation.original_text,
        ),
        intent_interpreter=lambda text: IntentParseResult(
            interpretation=interpretation,
            duration_ms=30.0,
        ),
        policy_engine=policy_engine
        or PolicyEngine(
            build_default_registry()
        ),
        execution_service=execution_service
        or _FakeExecutionService(),
    )


def _voice_result(
    transcript: str,
) -> VoiceResult:
    return VoiceResult(
        success=True,
        message="Comando reconocido.",
        transcript=transcript,
        timings_ms={
            "audio_capture": 5000.0,
            "transcription": 120.0,
        },
    )


def _open_application_interpretation(
    text: str,
    application_name: str,
) -> IntentInterpretation:
    return IntentInterpretation(
        original_text=text,
        normalized_text=text.casefold(),
        intent=IntentType.OPEN_APPLICATION,
        application_name=application_name,
    )


class _FakeExecutionService:
    def __init__(
        self,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.error = error

    def execute(
        self,
        request,
        decision,
    ) -> ToolResult:
        self.calls.append(
            request.tool_name,
        )

        if self.error is not None:
            raise self.error

        return ToolResult(
            success=True,
            tool_name=request.tool_name,
            message="fake execution",
            data={},
            duration_ms=3.0,
        )


class _StaticPolicyEngine:
    def __init__(
        self,
        decision: PolicyDecision,
    ) -> None:
        self.decision = decision

    def evaluate(
        self,
        action_request,
    ) -> PolicyDecision:
        return self.decision
