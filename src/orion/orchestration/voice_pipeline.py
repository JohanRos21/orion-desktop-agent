from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter

from orion.config import settings
from orion.execution.service import ExecutionService
from orion.llm.exceptions import (
    OllamaConfigurationError,
    OllamaError,
)
from orion.llm.intent_parser import interpret_intent
from orion.llm.models import (
    IntentInterpretation,
    IntentParseResult,
    IntentType,
)
from orion.models import ToolResult
from orion.orchestration.action_router import route_intent_to_action
from orion.policy.engine import PolicyEngine
from orion.policy.models import (
    PolicyDecision,
    PolicyDecisionType,
)
from orion.tools.contracts import (
    ActionRequest,
)
from orion.tools.default_registry import build_default_registry
from orion.voice import VoiceResult, listen_once


LOCAL_VOICE_SOURCE = "local_voice"

TIMING_KEYS = (
    "capture",
    "transcription",
    "llm",
    "routing",
    "policy",
    "execution",
    "total",
)

AUDIT_LOG: list["VoicePipelineAuditRecord"] = []


@dataclass(frozen=True, slots=True)
class VoicePipelineAuditRecord:
    timestamp: str
    capture_mode: str
    transcript: str
    normalized_text: str
    intent: str | None
    tool_name: str | None
    arguments: dict[str, object]
    risk: str | None
    decision: str | None
    execution_enabled: bool
    execution_attempted: bool
    execution_success: bool
    result_message: str
    capture_ms: float
    transcription_ms: float
    llm_ms: float
    policy_ms: float
    execution_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class VoicePipelineResult:
    transcript: str
    interpretation: IntentInterpretation | None
    action_request: ActionRequest | None
    policy_decision: PolicyDecision | None
    tool_result: ToolResult | None
    execution_enabled: bool
    timings_ms: dict[str, float]
    success: bool
    message: str
    execution_attempted: bool = False
    audit_record: VoicePipelineAuditRecord | None = None
    error_stage: str | None = None


class VoicePipeline:
    def __init__(
        self,
        voice_listener: Callable[[], VoiceResult] = listen_once,
        intent_interpreter: Callable[[str], IntentParseResult] = interpret_intent,
        policy_engine: PolicyEngine | None = None,
        execution_service: ExecutionService | None = None,
        action_router: Callable[
            [IntentInterpretation, str],
            ActionRequest | None,
        ] = route_intent_to_action,
    ) -> None:
        registry = build_default_registry()

        self.voice_listener = voice_listener
        self.intent_interpreter = intent_interpreter
        self.policy_engine = policy_engine or PolicyEngine(
            registry=registry,
        )
        self.execution_service = execution_service or ExecutionService(
            registry=registry,
        )
        self.action_router = action_router

    def run(
        self,
        execution_enabled: bool = False,
    ) -> VoicePipelineResult:
        total_started_at = perf_counter()
        timings = _empty_timings()
        capture_mode = settings.VOICE_CAPTURE_MODE

        try:
            voice_result = self.voice_listener()
        except Exception as error:
            timings["total"] = _milliseconds(
                total_started_at,
            )
            return self._finalize(
                transcript="",
                interpretation=None,
                action_request=None,
                policy_decision=None,
                tool_result=None,
                execution_enabled=execution_enabled,
                execution_attempted=False,
                timings=timings,
                capture_mode=capture_mode,
                success=False,
                message=f"Fallo la captura o transcripcion: {error}",
                error_stage="capture_transcription",
            )

        _merge_voice_timings(
            target=timings,
            voice_timings=voice_result.timings_ms,
        )

        transcript = (
            voice_result.transcript or ""
        ).strip()

        if not voice_result.success:
            timings["total"] = _milliseconds(
                total_started_at,
            )
            return self._finalize(
                transcript=transcript,
                interpretation=None,
                action_request=None,
                policy_decision=None,
                tool_result=None,
                execution_enabled=execution_enabled,
                execution_attempted=False,
                timings=timings,
                capture_mode=capture_mode,
                success=False,
                message=voice_result.message,
                error_stage="capture_transcription",
            )

        if not transcript:
            timings["total"] = _milliseconds(
                total_started_at,
            )
            return self._finalize(
                transcript="",
                interpretation=None,
                action_request=None,
                policy_decision=None,
                tool_result=None,
                execution_enabled=execution_enabled,
                execution_attempted=False,
                timings=timings,
                capture_mode=capture_mode,
                success=False,
                message=(
                    "La transcripcion esta vacia; "
                    "no ejecutare ninguna accion."
                ),
                error_stage="capture_transcription",
            )

        llm_started_at = perf_counter()

        try:
            llm_result = self.intent_interpreter(
                transcript,
            )
        except OllamaError as error:
            timings["llm"] = _milliseconds(
                llm_started_at,
            )
            timings["total"] = _milliseconds(
                total_started_at,
            )
            return self._finalize(
                transcript=transcript,
                interpretation=None,
                action_request=None,
                policy_decision=None,
                tool_result=None,
                execution_enabled=execution_enabled,
                execution_attempted=False,
                timings=timings,
                capture_mode=capture_mode,
                success=False,
                message=f"Fallo Ollama: {error}",
                error_stage=(
                    "configuration"
                    if isinstance(
                        error,
                        OllamaConfigurationError,
                    )
                    else "llm"
                ),
            )
        except Exception as error:
            timings["llm"] = _milliseconds(
                llm_started_at,
            )
            timings["total"] = _milliseconds(
                total_started_at,
            )
            return self._finalize(
                transcript=transcript,
                interpretation=None,
                action_request=None,
                policy_decision=None,
                tool_result=None,
                execution_enabled=execution_enabled,
                execution_attempted=False,
                timings=timings,
                capture_mode=capture_mode,
                success=False,
                message=f"Fallo la interpretacion: {error}",
                error_stage="application",
            )

        interpretation = llm_result.interpretation
        timings["llm"] = llm_result.duration_ms

        routing_started_at = perf_counter()
        action_request = self.action_router(
            interpretation,
            LOCAL_VOICE_SOURCE,
        )
        timings["routing"] = _milliseconds(
            routing_started_at,
        )

        policy_decision: PolicyDecision | None = None
        tool_result: ToolResult | None = None
        execution_attempted = False

        if action_request is not None:
            policy_decision = self.policy_engine.evaluate(
                action_request,
            )
            timings["policy"] = policy_decision.evaluation_ms

            if (
                execution_enabled
                and policy_decision.decision is PolicyDecisionType.ALLOW
            ):
                execution_attempted = True
                execution_started_at = perf_counter()

                try:
                    tool_result = self.execution_service.execute(
                        request=action_request,
                        decision=policy_decision,
                    )
                    timings["execution"] = tool_result.duration_ms
                except Exception as error:
                    execution_ms = _milliseconds(
                        execution_started_at,
                    )
                    tool_result = ToolResult(
                        success=False,
                        tool_name=action_request.tool_name,
                        message=(
                            "fallo controlado del executor: "
                            f"{error}"
                        ),
                        data={
                            "arguments": action_request.arguments,
                        },
                        duration_ms=execution_ms,
                    )
                    timings["execution"] = execution_ms

        timings["total"] = _milliseconds(
            total_started_at,
        )

        return self._finalize(
            transcript=transcript,
            interpretation=interpretation,
            action_request=action_request,
            policy_decision=policy_decision,
            tool_result=tool_result,
            execution_enabled=execution_enabled,
            execution_attempted=execution_attempted,
            timings=timings,
            capture_mode=capture_mode,
            success=_result_success(
                action_request=action_request,
                policy_decision=policy_decision,
                tool_result=tool_result,
            ),
            message=_result_message(
                interpretation=interpretation,
                action_request=action_request,
                policy_decision=policy_decision,
                tool_result=tool_result,
                execution_enabled=execution_enabled,
            ),
            error_stage=_result_error_stage(
                tool_result=tool_result,
            ),
        )

    def _finalize(
        self,
        transcript: str,
        interpretation: IntentInterpretation | None,
        action_request: ActionRequest | None,
        policy_decision: PolicyDecision | None,
        tool_result: ToolResult | None,
        execution_enabled: bool,
        execution_attempted: bool,
        timings: dict[str, float],
        capture_mode: str,
        success: bool,
        message: str,
        error_stage: str | None = None,
    ) -> VoicePipelineResult:
        audit_record = _build_audit_record(
            capture_mode=capture_mode,
            transcript=transcript,
            interpretation=interpretation,
            action_request=action_request,
            policy_decision=policy_decision,
            tool_result=tool_result,
            execution_enabled=execution_enabled,
            execution_attempted=execution_attempted,
            timings=timings,
            message=message,
        )
        AUDIT_LOG.append(
            audit_record,
        )

        return VoicePipelineResult(
            transcript=transcript,
            interpretation=interpretation,
            action_request=action_request,
            policy_decision=policy_decision,
            tool_result=tool_result,
            execution_enabled=execution_enabled,
            timings_ms=dict(timings),
            success=success,
            message=message,
            execution_attempted=execution_attempted,
            audit_record=audit_record,
            error_stage=error_stage,
        )


def _build_audit_record(
    capture_mode: str,
    transcript: str,
    interpretation: IntentInterpretation | None,
    action_request: ActionRequest | None,
    policy_decision: PolicyDecision | None,
    tool_result: ToolResult | None,
    execution_enabled: bool,
    execution_attempted: bool,
    timings: dict[str, float],
    message: str,
) -> VoicePipelineAuditRecord:
    return VoicePipelineAuditRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        capture_mode=capture_mode,
        transcript=transcript,
        normalized_text=(
            interpretation.normalized_text
            if interpretation is not None
            else ""
        ),
        intent=(
            interpretation.intent.value
            if interpretation is not None
            else None
        ),
        tool_name=(
            action_request.tool_name
            if action_request is not None
            else None
        ),
        arguments=(
            dict(action_request.arguments)
            if action_request is not None
            else {}
        ),
        risk=(
            policy_decision.risk_level.value
            if policy_decision is not None
            else None
        ),
        decision=(
            policy_decision.decision.value
            if policy_decision is not None
            else None
        ),
        execution_enabled=execution_enabled,
        execution_attempted=execution_attempted,
        execution_success=(
            tool_result.success
            if tool_result is not None
            else False
        ),
        result_message=message,
        capture_ms=timings["capture"],
        transcription_ms=timings["transcription"],
        llm_ms=timings["llm"],
        policy_ms=timings["policy"],
        execution_ms=timings["execution"],
        total_ms=timings["total"],
    )


def _empty_timings() -> dict[str, float]:
    return {
        key: 0.0
        for key in TIMING_KEYS
    }


def _merge_voice_timings(
    target: dict[str, float],
    voice_timings: dict[str, float],
) -> None:
    target["capture"] = _extract_capture_ms(
        voice_timings,
    )
    target["transcription"] = voice_timings.get(
        "transcription",
        0.0,
    )


def _extract_capture_ms(
    voice_timings: dict[str, float],
) -> float:
    for key in (
        "audio_capture",
        "vad_total",
        "speech_capture",
        "capture",
    ):
        if key in voice_timings:
            return voice_timings[key]

    return 0.0


def _result_success(
    action_request: ActionRequest | None,
    policy_decision: PolicyDecision | None,
    tool_result: ToolResult | None,
) -> bool:
    if action_request is None:
        return True

    if policy_decision is None:
        return False

    if policy_decision.decision is not PolicyDecisionType.ALLOW:
        return True

    if tool_result is None:
        return True

    return tool_result.success


def _result_message(
    interpretation: IntentInterpretation,
    action_request: ActionRequest | None,
    policy_decision: PolicyDecision | None,
    tool_result: ToolResult | None,
    execution_enabled: bool,
) -> str:
    if tool_result is not None:
        return tool_result.message

    if interpretation.intent is IntentType.CONVERSATION:
        return (
            interpretation.assistant_reply
            or "Conversacion recibida; no se requiere herramienta."
        )

    if interpretation.intent is IntentType.UNKNOWN:
        return (
            interpretation.clarification_question
            or "No pude determinar una accion segura."
        )

    if action_request is None:
        return "No se creo ninguna accion ejecutable."

    if policy_decision is None:
        return "No hay decision de politica."

    if policy_decision.decision is PolicyDecisionType.BLOCK:
        return f"Bloqueado por politica: {policy_decision.reason}"

    if policy_decision.decision is PolicyDecisionType.CONFIRM:
        return (
            "La accion requiere confirmacion; "
            "la demo de voz no la ejecuta."
        )

    if not execution_enabled:
        return "Simulacion: ejecucion real deshabilitada."

    return "Ejecucion no intentada."


def _result_error_stage(
    tool_result: ToolResult | None,
) -> str | None:
    if tool_result is not None and not tool_result.success:
        return "execution"

    return None


def _milliseconds(
    started_at: float,
) -> float:
    return round(
        (perf_counter() - started_at) * 1000,
        2,
    )
