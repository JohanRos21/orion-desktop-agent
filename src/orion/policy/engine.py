from __future__ import annotations

from time import perf_counter

from orion.policy.models import (
    PolicyDecision,
    PolicyDecisionType,
)
from orion.tools.applications import (
    ApplicationValidationError,
    validate_application_request,
)
from orion.tools.contracts import (
    ALLOWED_ACTION_SOURCES,
    ActionRequest,
    RiskLevel,
)
from orion.tools.registry import ToolRegistry


class PolicyEngine:
    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        self.registry = registry

    def evaluate(
        self,
        action_request: ActionRequest,
    ) -> PolicyDecision:
        started_at = perf_counter()

        decision = self._evaluate(
            action_request=action_request,
        )

        return PolicyDecision(
            decision=decision.decision,
            risk_level=decision.risk_level,
            reason=decision.reason,
            tool_name=decision.tool_name,
            arguments=decision.arguments,
            evaluation_ms=round(
                (perf_counter() - started_at) * 1000,
                2,
            ),
        )

    def _evaluate(
        self,
        action_request: ActionRequest,
    ) -> PolicyDecision:
        tool_definition = self.registry.get(
            action_request.tool_name
        )

        if tool_definition is None:
            return _block(
                reason="herramienta no registrada",
                tool_name=action_request.tool_name,
                arguments=action_request.arguments,
            )

        if action_request.source not in ALLOWED_ACTION_SOURCES:
            return _block(
                reason="fuente no permitida",
                tool_name=tool_definition.name,
                arguments=action_request.arguments,
                risk_level=tool_definition.risk_level,
            )

        if tool_definition.risk_level is RiskLevel.BLOCKED:
            return _block(
                reason="herramienta bloqueada por politica",
                tool_name=tool_definition.name,
                arguments=action_request.arguments,
                risk_level=tool_definition.risk_level,
            )

        missing_arguments = tuple(
            argument
            for argument in tool_definition.required_arguments
            if argument not in action_request.arguments
        )

        if missing_arguments:
            return _block(
                reason=(
                    "faltan argumentos requeridos: "
                    + ", ".join(missing_arguments)
                ),
                tool_name=tool_definition.name,
                arguments=action_request.arguments,
                risk_level=tool_definition.risk_level,
            )

        if tool_definition.name == "open_application":
            validation_error = _validate_open_application(
                arguments=action_request.arguments,
                original_text=action_request.original_text,
            )

            if validation_error is not None:
                return _block(
                    reason=validation_error,
                    tool_name=tool_definition.name,
                    arguments=action_request.arguments,
                    risk_level=tool_definition.risk_level,
                )

        if tool_definition.risk_level is RiskLevel.LOW:
            return PolicyDecision(
                decision=PolicyDecisionType.ALLOW,
                risk_level=tool_definition.risk_level,
                reason=(
                    "herramienta registrada y argumentos validos"
                ),
                tool_name=tool_definition.name,
                arguments=action_request.arguments,
                evaluation_ms=0.0,
            )

        if tool_definition.risk_level in {
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
        }:
            return PolicyDecision(
                decision=PolicyDecisionType.CONFIRM,
                risk_level=tool_definition.risk_level,
                reason=(
                    "la herramienta requiere confirmacion"
                ),
                tool_name=tool_definition.name,
                arguments=action_request.arguments,
                evaluation_ms=0.0,
            )

        return _block(
            reason="nivel de riesgo no permitido",
            tool_name=tool_definition.name,
            arguments=action_request.arguments,
            risk_level=tool_definition.risk_level,
        )


def _validate_open_application(
    arguments: dict[str, object],
    original_text: str,
) -> str | None:
    try:
        validate_application_request(
            application_name=arguments.get(
                "application_name"
            ),
            original_text=original_text,
        )
    except ApplicationValidationError as error:
        return str(error)

    return None


def _block(
    reason: str,
    tool_name: str | None,
    arguments: dict[str, object],
    risk_level: RiskLevel = RiskLevel.BLOCKED,
) -> PolicyDecision:
    return PolicyDecision(
        decision=PolicyDecisionType.BLOCK,
        risk_level=risk_level,
        reason=reason,
        tool_name=tool_name,
        arguments=arguments,
        evaluation_ms=0.0,
    )
