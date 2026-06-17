from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

from orion.execution.models import ExecutionAuditRecord
from orion.execution.service import ExecutionService
from orion.llm.exceptions import OllamaError
from orion.llm.intent_parser import interpret_intent
from orion.llm.models import IntentInterpretation
from orion.models import ToolResult
from orion.orchestration.action_router import route_intent_to_action
from orion.policy.engine import PolicyEngine
from orion.policy.models import (
    PolicyDecision,
    PolicyDecisionType,
)
from orion.tools.applications import (
    ApplicationValidationError,
    validate_application_request,
)
from orion.tools.contracts import ActionRequest
from orion.tools.default_registry import build_default_registry


EXIT_COMMANDS = {
    "salir",
    "exit",
    "quit",
}

AUDIT_LOG: list["ExecutionDemoAuditRecord"] = []


@dataclass(frozen=True, slots=True)
class ExecutionDemoAuditRecord:
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
    execution_requested: bool
    execution_enabled: bool
    execution_attempted: bool
    execution_success: bool
    execution_ms: float
    result_message: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo de ejecucion controlada de ORION."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Habilita ejecucion real de herramientas permitidas.",
    )
    args = parser.parse_args()

    registry = build_default_registry()
    policy_engine = PolicyEngine(
        registry=registry,
    )
    execution_service = ExecutionService(
        registry=registry,
    )

    print("ORION Execution Demo")
    print(
        "Ejecucion real habilitada: "
        f"{args.execute}"
    )
    print("Usa salir, exit o quit.")

    while True:
        try:
            text = input("\nTexto > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nORION [EXEC] > Cerrando demo.")
            break

        if not text:
            continue

        if text.casefold() in EXIT_COMMANDS:
            print("ORION [EXEC] > Cerrando demo.")
            break

        try:
            handle_text(
                text=text,
                execution_enabled=args.execute,
                policy_engine=policy_engine,
                execution_service=execution_service,
            )
        except OllamaError as error:
            print(
                "ORION [LLM ERROR] > "
                f"{error}"
            )


def handle_text(
    text: str,
    execution_enabled: bool,
    policy_engine: PolicyEngine,
    execution_service: ExecutionService,
) -> ExecutionDemoAuditRecord:
    llm_result = interpret_intent(
        text,
    )
    interpretation = llm_result.interpretation

    action_request = route_intent_to_action(
        interpretation=interpretation,
        source="local_text",
    )

    policy_decision: PolicyDecision | None = None
    result: ToolResult | None = None
    execution_attempted = False

    if action_request is not None:
        policy_decision = policy_engine.evaluate(
            action_request
        )

        if (
            execution_enabled
            and policy_decision.decision is PolicyDecisionType.ALLOW
        ):
            execution_attempted = True
            result = execution_service.execute(
                request=action_request,
                decision=policy_decision,
            )

    audit_record = build_execution_audit_record(
        interpretation=interpretation,
        action_request=action_request,
        policy_decision=policy_decision,
        result=result,
        llm_ms=llm_result.duration_ms,
        execution_enabled=execution_enabled,
        execution_attempted=execution_attempted,
    )
    AUDIT_LOG.append(
        audit_record
    )

    _print_summary(
        interpretation=interpretation,
        action_request=action_request,
        policy_decision=policy_decision,
        audit_record=audit_record,
    )

    return audit_record


def build_execution_audit_record(
    interpretation: IntentInterpretation,
    action_request: ActionRequest | None,
    policy_decision: PolicyDecision | None,
    result: ToolResult | None,
    llm_ms: float,
    execution_enabled: bool,
    execution_attempted: bool,
) -> ExecutionDemoAuditRecord:
    execution_audit = ExecutionAuditRecord(
        execution_requested=action_request is not None,
        execution_enabled=execution_enabled,
        execution_attempted=execution_attempted,
        execution_success=(
            result.success
            if result is not None
            else False
        ),
        execution_ms=(
            result.duration_ms
            if result is not None
            else 0.0
        ),
        result_message=(
            result.message
            if result is not None
            else _default_result_message(
                action_request=action_request,
                policy_decision=policy_decision,
                execution_enabled=execution_enabled,
            )
        ),
    )

    return ExecutionDemoAuditRecord(
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
        execution_requested=execution_audit.execution_requested,
        execution_enabled=execution_audit.execution_enabled,
        execution_attempted=execution_audit.execution_attempted,
        execution_success=execution_audit.execution_success,
        execution_ms=execution_audit.execution_ms,
        result_message=execution_audit.result_message,
    )


def _default_result_message(
    action_request: ActionRequest | None,
    policy_decision: PolicyDecision | None,
    execution_enabled: bool,
) -> str:
    if action_request is None:
        return "no se requiere herramienta"

    if policy_decision is None:
        return "no hay decision de politica"

    if policy_decision.decision is not PolicyDecisionType.ALLOW:
        return "la politica bloqueo la ejecucion"

    if not execution_enabled:
        return "simulacion: ejecucion real deshabilitada"

    return "no se intento ejecutar"


def _print_summary(
    interpretation: IntentInterpretation,
    action_request: ActionRequest | None,
    policy_decision: PolicyDecision | None,
    audit_record: ExecutionDemoAuditRecord,
) -> None:
    print("\nIntent:")
    print(f"  {interpretation.intent.value}")

    if interpretation.clarification_question:
        print("\nAclaracion:")
        print(f"  {interpretation.clarification_question}")

    if action_request is None:
        print("\nNo se requiere herramienta.")
        _print_execution(audit_record)
        return

    print("\nHerramienta:")
    print(f"  {action_request.tool_name}")
    print("\nAplicacion resuelta:")
    print(
        "  "
        + _resolve_application_label(
            action_request
        )
    )

    if policy_decision is not None:
        print("\nDecision de politica:")
        print(f"  {policy_decision.decision.value}")
        print("\nRiesgo:")
        print(f"  {policy_decision.risk_level.value}")
        print("\nRazon:")
        print(f"  {policy_decision.reason}")

    _print_execution(audit_record)


def _print_execution(
    audit_record: ExecutionDemoAuditRecord,
) -> None:
    print("\nEjecucion real habilitada:")
    print(f"  {audit_record.execution_enabled}")
    print("\nEjecucion intentada:")
    print(f"  {audit_record.execution_attempted}")
    print("\nResultado:")
    print(f"  {audit_record.result_message}")
    print("\nTiempos:")
    print(f"  llm_ms = {audit_record.llm_ms:.2f}")
    print(f"  policy_ms = {audit_record.policy_ms:.2f}")
    print(f"  execution_ms = {audit_record.execution_ms:.2f}")


def _resolve_application_label(
    action_request: ActionRequest,
) -> str:
    application_name = action_request.arguments.get(
        "application_name"
    )

    try:
        application = validate_application_request(
            application_name=application_name,
            original_text=action_request.original_text,
        )
    except ApplicationValidationError as error:
        return f"no resuelta ({error})"

    return (
        f"{application.display_name} "
        f"({application.executable})"
    )


if __name__ == "__main__":
    main()
