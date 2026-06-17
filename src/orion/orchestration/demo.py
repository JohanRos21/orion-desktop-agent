from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from orion.llm.exceptions import OllamaError
from orion.llm.intent_parser import interpret_intent
from orion.llm.models import IntentInterpretation
from orion.orchestration.action_router import route_intent_to_action
from orion.policy.engine import PolicyEngine
from orion.policy.models import PolicyDecision
from orion.tools.contracts import ActionRequest
from orion.tools.default_registry import build_default_registry


EXIT_COMMANDS = {
    "salir",
    "exit",
    "quit",
}

AUDIT_LOG: list["AuditRecord"] = []


@dataclass(frozen=True, slots=True)
class AuditRecord:
    timestamp: str
    original_text: str
    intent: str
    tool_name: str | None
    arguments: dict[str, object]
    source: str
    risk: str | None
    decision: str | None
    reason: str
    llm_ms: float
    policy_ms: float


def main() -> None:
    registry = build_default_registry()
    policy_engine = PolicyEngine(
        registry=registry,
    )

    print("ORION Orchestration Demo")
    print("No ejecuta herramientas. Usa salir, exit o quit.")

    while True:
        try:
            text = input("\nTexto > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nORION [ORCH] > Cerrando demo.")
            break

        if not text:
            continue

        if text.casefold() in EXIT_COMMANDS:
            print("ORION [ORCH] > Cerrando demo.")
            break

        try:
            _handle_text(
                text=text,
                policy_engine=policy_engine,
            )
        except OllamaError as error:
            print(
                "ORION [LLM ERROR] > "
                f"{error}"
            )


def _handle_text(
    text: str,
    policy_engine: PolicyEngine,
) -> None:
    llm_result = interpret_intent(
        text,
    )
    interpretation = llm_result.interpretation

    action_request = route_intent_to_action(
        interpretation=interpretation,
        source="local_text",
    )

    policy_decision: PolicyDecision | None = None

    if action_request is not None:
        policy_decision = policy_engine.evaluate(
            action_request
        )

    audit_record = build_audit_record(
        interpretation=interpretation,
        action_request=action_request,
        policy_decision=policy_decision,
        llm_ms=llm_result.duration_ms,
    )
    AUDIT_LOG.append(
        audit_record
    )

    _print_interpretation(
        interpretation
    )
    _print_action(
        action_request
    )
    _print_policy(
        policy_decision=policy_decision,
        interpretation=interpretation,
    )
    _print_timings(
        audit_record
    )


def build_audit_record(
    interpretation: IntentInterpretation,
    action_request: ActionRequest | None,
    policy_decision: PolicyDecision | None,
    llm_ms: float,
) -> AuditRecord:
    return AuditRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        original_text=interpretation.original_text,
        intent=interpretation.intent.value,
        tool_name=(
            action_request.tool_name
            if action_request is not None
            else None
        ),
        arguments=(
            action_request.arguments
            if action_request is not None
            else {}
        ),
        source=(
            action_request.source
            if action_request is not None
            else "local_text"
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
        reason=(
            policy_decision.reason
            if policy_decision is not None
            else "no se requiere herramienta"
        ),
        llm_ms=llm_ms,
        policy_ms=(
            policy_decision.evaluation_ms
            if policy_decision is not None
            else 0.0
        ),
    )


def _print_interpretation(
    interpretation: IntentInterpretation,
) -> None:
    print("\nIntentInterpretation:")
    print(
        f"  original_text = {interpretation.original_text}"
    )
    print(
        f"  normalized_text = {interpretation.normalized_text}"
    )
    print(
        f"  intent = {interpretation.intent.value}"
    )
    print(
        f"  application_name = {interpretation.application_name or ''}"
    )
    print(
        f"  needs_clarification = {interpretation.needs_clarification}"
    )
    print(
        "  clarification_question = "
        f"{interpretation.clarification_question or ''}"
    )
    print(
        f"  assistant_reply = {interpretation.assistant_reply or ''}"
    )


def _print_action(
    action_request: ActionRequest | None,
) -> None:
    if action_request is None:
        print("\nActionRequest:")
        print("  No se creo ninguna accion.")
        return

    print("\nActionRequest:")
    print(
        f"  tool_name = {action_request.tool_name}"
    )
    print(
        "  application_name = "
        f"{action_request.arguments.get('application_name', '')}"
    )
    print(
        f"  source = {action_request.source}"
    )


def _print_policy(
    policy_decision: PolicyDecision | None,
    interpretation: IntentInterpretation,
) -> None:
    if policy_decision is None:
        if interpretation.assistant_reply:
            print(
                "\nRespuesta conversacional:"
            )
            print(
                f"  {interpretation.assistant_reply}"
            )

        if interpretation.clarification_question:
            print("\nAclaracion:")
            print(
                f"  {interpretation.clarification_question}"
            )

        print("\nPolicy:")
        print("  No se requiere herramienta.")
        return

    print("\nPolicy:")
    print(
        f"  decision = {policy_decision.decision.value}"
    )
    print(
        f"  risk = {policy_decision.risk_level.value}"
    )
    print(
        f"  reason = {policy_decision.reason}"
    )
    print(
        f"  policy_ms = {policy_decision.evaluation_ms:.2f}"
    )


def _print_timings(
    audit_record: AuditRecord,
) -> None:
    print("\nTiempos:")
    print(
        f"  llm_ms = {audit_record.llm_ms:.2f}"
    )
    print(
        f"  policy_ms = {audit_record.policy_ms:.2f}"
    )


if __name__ == "__main__":
    main()
