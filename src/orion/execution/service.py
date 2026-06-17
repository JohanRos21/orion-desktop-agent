from __future__ import annotations

from time import perf_counter

from orion.execution.exceptions import ToolExecutionError
from orion.models import ToolResult
from orion.policy.models import (
    PolicyDecision,
    PolicyDecisionType,
)
from orion.tools.applications import (
    ApplicationValidationError,
    validate_application_request,
)
from orion.tools.contracts import ActionRequest
from orion.tools.registry import ToolRegistry


class ExecutionService:
    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        self.registry = registry

    def execute(
        self,
        request: ActionRequest,
        decision: PolicyDecision,
    ) -> ToolResult:
        started_at = perf_counter()

        rejection_reason = self._find_rejection_reason(
            request=request,
            decision=decision,
        )

        if rejection_reason is not None:
            return _failure(
                tool_name=request.tool_name,
                message=rejection_reason,
                duration_ms=_milliseconds(started_at),
                data={
                    "arguments": request.arguments,
                },
            )

        tool_definition = self.registry.get(
            request.tool_name
        )

        if tool_definition is None or tool_definition.executor is None:
            return _failure(
                tool_name=request.tool_name,
                message="herramienta no ejecutable",
                duration_ms=_milliseconds(started_at),
                data={
                    "arguments": request.arguments,
                },
            )

        safe_arguments = self._build_safe_arguments(
            request
        )

        try:
            executor_result = tool_definition.executor(
                **safe_arguments
            )
        except Exception as error:
            execution_error = ToolExecutionError(
                str(error)
            )
            return _failure(
                tool_name=tool_definition.name,
                message=(
                    "fallo el executor registrado: "
                    f"{execution_error}"
                ),
                duration_ms=_milliseconds(started_at),
                data={
                    "arguments": safe_arguments,
                },
            )

        duration_ms = _milliseconds(started_at)

        return ToolResult(
            success=executor_result.success,
            tool_name=tool_definition.name,
            message=executor_result.message,
            data={
                **executor_result.data,
                "arguments": safe_arguments,
            },
            duration_ms=duration_ms,
        )

    def _find_rejection_reason(
        self,
        request: ActionRequest,
        decision: PolicyDecision,
    ) -> str | None:
        if decision.decision is not PolicyDecisionType.ALLOW:
            return (
                "la decision de politica no permite ejecucion"
            )

        if request.tool_name != decision.tool_name:
            return (
                "la herramienta de la solicitud no coincide "
                "con la decision"
            )

        if request.arguments != decision.arguments:
            return (
                "los argumentos de la solicitud no coinciden "
                "con la decision"
            )

        tool_definition = self.registry.get(
            request.tool_name
        )

        if tool_definition is None:
            return "herramienta no registrada"

        if tool_definition.executor is None:
            return "herramienta sin executor"

        missing_arguments = tuple(
            argument
            for argument in tool_definition.required_arguments
            if argument not in request.arguments
        )

        if missing_arguments:
            return (
                "faltan argumentos requeridos: "
                + ", ".join(missing_arguments)
            )

        try:
            self._build_safe_arguments(
                request
            )
        except ApplicationValidationError as error:
            return str(error)

        return None

    def _build_safe_arguments(
        self,
        request: ActionRequest,
    ) -> dict[str, object]:
        if request.tool_name != "open_application":
            return dict(
                request.arguments
            )

        application = validate_application_request(
            application_name=request.arguments.get(
                "application_name"
            ),
            original_text=request.original_text,
        )

        return {
            "application_name": application.canonical_name,
        }


def _failure(
    tool_name: str,
    message: str,
    duration_ms: float,
    data: dict[str, object] | None = None,
) -> ToolResult:
    return ToolResult(
        success=False,
        tool_name=tool_name,
        message=message,
        data=data or {},
        duration_ms=duration_ms,
    )


def _milliseconds(
    started_at: float,
) -> float:
    return round(
        (perf_counter() - started_at) * 1000,
        2,
    )
