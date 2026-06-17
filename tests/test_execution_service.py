from orion.execution.service import ExecutionService
from orion.models import ToolResult
from orion.policy.models import (
    PolicyDecision,
    PolicyDecisionType,
)
from orion.tools.contracts import (
    ActionRequest,
    RiskLevel,
    ToolDefinition,
)
from orion.tools.registry import ToolRegistry


def test_allow_valid_request_executes_once() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)
    request = _open_application_request("calculator")
    decision = _allow_decision(request)

    result = ExecutionService(registry).execute(
        request=request,
        decision=decision,
    )

    assert result.success is True
    assert result.tool_name == "open_application"
    assert result.duration_ms >= 0
    assert calls == [{"application_name": "calculadora"}]


def test_block_decision_never_executes() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)
    request = _open_application_request("calculadora")

    result = ExecutionService(registry).execute(
        request=request,
        decision=_decision(
            request,
            PolicyDecisionType.BLOCK,
        ),
    )

    assert result.success is False
    assert calls == []


def test_confirm_decision_never_executes() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)
    request = _open_application_request("calculadora")

    result = ExecutionService(registry).execute(
        request=request,
        decision=_decision(
            request,
            PolicyDecisionType.CONFIRM,
        ),
    )

    assert result.success is False
    assert calls == []


def test_mismatched_tool_name_never_executes() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)
    request = _open_application_request("calculadora")
    decision = _allow_decision(request)
    decision = PolicyDecision(
        decision=decision.decision,
        risk_level=decision.risk_level,
        reason=decision.reason,
        tool_name="other_tool",
        arguments=decision.arguments,
        evaluation_ms=0.0,
    )

    result = ExecutionService(registry).execute(
        request=request,
        decision=decision,
    )

    assert result.success is False
    assert calls == []


def test_mismatched_arguments_never_execute() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)
    request = _open_application_request("calculadora")
    decision = _allow_decision(request)
    decision = PolicyDecision(
        decision=decision.decision,
        risk_level=decision.risk_level,
        reason=decision.reason,
        tool_name=decision.tool_name,
        arguments={"application_name": "notepad"},
        evaluation_ms=0.0,
    )

    result = ExecutionService(registry).execute(
        request=request,
        decision=decision,
    )

    assert result.success is False
    assert calls == []


def test_missing_tool_never_executes() -> None:
    request = _open_application_request("calculadora")

    result = ExecutionService(ToolRegistry()).execute(
        request=request,
        decision=_allow_decision(request),
    )

    assert result.success is False
    assert "no registrada" in result.message


def test_tool_without_executor_never_executes() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="open_application",
            description="Sin executor.",
            risk_level=RiskLevel.LOW,
            required_arguments=("application_name",),
            executor=None,
        )
    )
    request = _open_application_request("calculadora")

    result = ExecutionService(registry).execute(
        request=request,
        decision=_allow_decision(request),
    )

    assert result.success is False
    assert "sin executor" in result.message


def test_missing_required_argument_never_executes() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)
    request = ActionRequest(
        tool_name="open_application",
        arguments={},
        original_text="abre",
        source="local_text",
    )

    result = ExecutionService(registry).execute(
        request=request,
        decision=_allow_decision(request),
    )

    assert result.success is False
    assert calls == []


def test_expected_executor_exception_returns_failed_result() -> None:
    def failing_executor(application_name: str) -> ToolResult:
        raise OSError("boom")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="open_application",
            description="Falla.",
            risk_level=RiskLevel.LOW,
            required_arguments=("application_name",),
            executor=failing_executor,
        )
    )
    request = _open_application_request("calculadora")

    result = ExecutionService(registry).execute(
        request=request,
        decision=_allow_decision(request),
    )

    assert result.success is False
    assert "fallo el executor" in result.message
    assert result.duration_ms >= 0


def test_allowed_aliases_are_resolved_before_execution() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)

    for alias in ("calculadora", "calculator", "calc"):
        request = _open_application_request(alias)
        result = ExecutionService(registry).execute(
            request=request,
            decision=_allow_decision(request),
        )

        assert result.success is True

    assert calls == [
        {"application_name": "calculadora"},
        {"application_name": "calculadora"},
        {"application_name": "calculadora"},
    ]


def test_powershell_is_rejected_before_execution() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)
    request = _open_application_request("powershell")

    result = ExecutionService(registry).execute(
        request=request,
        decision=_allow_decision(request),
    )

    assert result.success is False
    assert calls == []


def test_exe_path_is_rejected_before_execution() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)
    request = _open_application_request("C:\\Temp\\tool.exe")

    result = ExecutionService(registry).execute(
        request=request,
        decision=_allow_decision(request),
    )

    assert result.success is False
    assert calls == []


def test_shell_operators_are_rejected_before_execution() -> None:
    calls: list[dict[str, object]] = []
    registry = _registry_with_executor(calls)
    request = _open_application_request(
        "calculadora && powershell"
    )

    result = ExecutionService(registry).execute(
        request=request,
        decision=_allow_decision(request),
    )

    assert result.success is False
    assert calls == []


def _registry_with_executor(
    calls: list[dict[str, object]],
) -> ToolRegistry:
    def fake_executor(application_name: str) -> ToolResult:
        calls.append(
            {
                "application_name": application_name,
            }
        )
        return ToolResult(
            success=True,
            tool_name="open_application",
            message=f"opened {application_name}",
            data={
                "application_name": application_name,
            },
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="open_application",
            description="Fake app opener.",
            risk_level=RiskLevel.LOW,
            required_arguments=("application_name",),
            executor=fake_executor,
        )
    )

    return registry


def _open_application_request(
    application_name: str,
) -> ActionRequest:
    return ActionRequest(
        tool_name="open_application",
        arguments={
            "application_name": application_name,
        },
        original_text=f"abre {application_name}",
        source="local_text",
    )


def _allow_decision(
    request: ActionRequest,
) -> PolicyDecision:
    return _decision(
        request,
        PolicyDecisionType.ALLOW,
    )


def _decision(
    request: ActionRequest,
    decision: PolicyDecisionType,
) -> PolicyDecision:
    return PolicyDecision(
        decision=decision,
        risk_level=RiskLevel.LOW,
        reason="test",
        tool_name=request.tool_name,
        arguments=request.arguments,
        evaluation_ms=0.0,
    )
