from orion.execution.service import ExecutionService
from orion.llm.models import (
    IntentInterpretation,
    IntentParseResult,
    IntentType,
)
from orion.models import ToolResult
from orion.orchestration import execution_demo
from orion.policy.engine import PolicyEngine
from orion.tools.default_registry import build_default_registry


def test_demo_without_execute_never_executes(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        execution_demo,
        "interpret_intent",
        lambda text: _llm_result(text, "calculadora"),
    )

    audit_record = execution_demo.handle_text(
        text="Abre la calculadora",
        execution_enabled=False,
        policy_engine=PolicyEngine(
            build_default_registry()
        ),
        execution_service=_fake_execution_service(calls),
    )

    assert audit_record.decision == "allow"
    assert audit_record.execution_requested is True
    assert audit_record.execution_enabled is False
    assert audit_record.execution_attempted is False
    assert calls == []


def test_demo_with_execute_runs_only_after_allow(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        execution_demo,
        "interpret_intent",
        lambda text: _llm_result(text, "calculadora"),
    )

    audit_record = execution_demo.handle_text(
        text="Abre la calculadora",
        execution_enabled=True,
        policy_engine=PolicyEngine(
            build_default_registry()
        ),
        execution_service=_fake_execution_service(calls),
    )

    assert audit_record.decision == "allow"
    assert audit_record.execution_attempted is True
    assert audit_record.execution_success is True
    assert calls == ["open_application"]


def test_demo_with_execute_does_not_run_blocked_action(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        execution_demo,
        "interpret_intent",
        lambda text: _llm_result(text, "powershell"),
    )

    audit_record = execution_demo.handle_text(
        text="Abre PowerShell",
        execution_enabled=True,
        policy_engine=PolicyEngine(
            build_default_registry()
        ),
        execution_service=_fake_execution_service(calls),
    )

    assert audit_record.decision == "block"
    assert audit_record.execution_attempted is False
    assert audit_record.execution_success is False
    assert calls == []


def test_execution_audit_records_result_fields(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        execution_demo,
        "interpret_intent",
        lambda text: _llm_result(text, "calculadora"),
    )

    audit_record = execution_demo.handle_text(
        text="Abre la calculadora",
        execution_enabled=True,
        policy_engine=PolicyEngine(
            build_default_registry()
        ),
        execution_service=_fake_execution_service(calls),
    )

    assert audit_record.execution_requested is True
    assert audit_record.execution_enabled is True
    assert audit_record.execution_attempted is True
    assert audit_record.execution_success is True
    assert audit_record.execution_ms >= 0
    assert audit_record.result_message == "fake execution"


class _FakeExecutionService(ExecutionService):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def execute(self, request, decision) -> ToolResult:
        self.calls.append(
            request.tool_name
        )
        return ToolResult(
            success=True,
            tool_name=request.tool_name,
            message="fake execution",
            data={},
            duration_ms=1.0,
        )


def _fake_execution_service(
    calls: list[str],
) -> ExecutionService:
    return _FakeExecutionService(calls)


def _llm_result(
    text: str,
    application_name: str,
) -> IntentParseResult:
    return IntentParseResult(
        interpretation=IntentInterpretation(
            original_text=text,
            normalized_text=text.casefold(),
            intent=IntentType.OPEN_APPLICATION,
            application_name=application_name,
        ),
        duration_ms=10.0,
    )
