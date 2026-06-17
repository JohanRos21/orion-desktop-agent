import pytest

from orion.models import ToolResult
from orion.policy.engine import PolicyEngine
from orion.policy.models import PolicyDecisionType
from orion.tools.contracts import (
    ActionRequest,
    RiskLevel,
    ToolDefinition,
)
from orion.tools.default_registry import build_default_registry
from orion.tools.registry import ToolRegistry


def test_unregistered_tool_is_blocked() -> None:
    decision = PolicyEngine(
        registry=ToolRegistry(),
    ).evaluate(
        ActionRequest(
            tool_name="missing_tool",
            arguments={},
            original_text="haz algo",
            source="local_text",
        )
    )

    assert decision.decision is PolicyDecisionType.BLOCK
    assert decision.reason == "herramienta no registrada"


def test_unknown_source_is_blocked() -> None:
    decision = PolicyEngine(
        registry=build_default_registry(),
    ).evaluate(
        ActionRequest(
            tool_name="open_application",
            arguments={
                "application_name": "calculadora",
            },
            original_text="abre calculadora",
            source="remote_api",
        )
    )

    assert decision.decision is PolicyDecisionType.BLOCK
    assert decision.reason == "fuente no permitida"


def test_missing_required_argument_is_blocked() -> None:
    decision = PolicyEngine(
        registry=build_default_registry(),
    ).evaluate(
        ActionRequest(
            tool_name="open_application",
            arguments={},
            original_text="abre",
            source="local_text",
        )
    )

    assert decision.decision is PolicyDecisionType.BLOCK
    assert "faltan argumentos requeridos" in decision.reason


def test_low_risk_valid_action_is_allowed() -> None:
    decision = PolicyEngine(
        registry=build_default_registry(),
    ).evaluate(
        ActionRequest(
            tool_name="open_application",
            arguments={
                "application_name": "calculadora",
            },
            original_text="abre calculadora",
            source="local_text",
        )
    )

    assert decision.decision is PolicyDecisionType.ALLOW
    assert decision.risk_level is RiskLevel.LOW
    assert decision.evaluation_ms >= 0


def test_medium_risk_requires_confirmation() -> None:
    decision = PolicyEngine(
        registry=_registry_with_tool(
            RiskLevel.MEDIUM,
        ),
    ).evaluate(
        _generic_action()
    )

    assert decision.decision is PolicyDecisionType.CONFIRM
    assert decision.risk_level is RiskLevel.MEDIUM


def test_high_risk_requires_confirmation() -> None:
    decision = PolicyEngine(
        registry=_registry_with_tool(
            RiskLevel.HIGH,
        ),
    ).evaluate(
        _generic_action()
    )

    assert decision.decision is PolicyDecisionType.CONFIRM
    assert decision.risk_level is RiskLevel.HIGH


def test_blocked_risk_is_blocked() -> None:
    decision = PolicyEngine(
        registry=_registry_with_tool(
            RiskLevel.BLOCKED,
        ),
    ).evaluate(
        _generic_action()
    )

    assert decision.decision is PolicyDecisionType.BLOCK
    assert decision.risk_level is RiskLevel.BLOCKED


def test_empty_application_name_is_blocked() -> None:
    decision = PolicyEngine(
        registry=build_default_registry(),
    ).evaluate(
        ActionRequest(
            tool_name="open_application",
            arguments={
                "application_name": " ",
            },
            original_text="abre",
            source="local_text",
        )
    )

    assert decision.decision is PolicyDecisionType.BLOCK
    assert "vacio" in decision.reason


def test_powershell_application_name_is_blocked() -> None:
    decision = PolicyEngine(
        registry=build_default_registry(),
    ).evaluate(
        ActionRequest(
            tool_name="open_application",
            arguments={
                "application_name": "powershell",
            },
            original_text="abre powershell",
            source="local_text",
        )
    )

    assert decision.decision is PolicyDecisionType.BLOCK
    assert "shell" in decision.reason


def test_shell_operator_application_name_is_blocked() -> None:
    decision = PolicyEngine(
        registry=build_default_registry(),
    ).evaluate(
        ActionRequest(
            tool_name="open_application",
            arguments={
                "application_name": "calculadora && borra archivos",
            },
            original_text="abre calculadora && borra archivos",
            source="local_text",
        )
    )

    assert decision.decision is PolicyDecisionType.BLOCK
    assert "shell" in decision.reason


def test_shell_operator_in_original_text_is_blocked() -> None:
    decision = PolicyEngine(
        registry=build_default_registry(),
    ).evaluate(
        ActionRequest(
            tool_name="open_application",
            arguments={
                "application_name": "calculadora",
            },
            original_text="abre calculadora && borra archivos",
            source="local_text",
        )
    )

    assert decision.decision is PolicyDecisionType.BLOCK
    assert "texto original" in decision.reason


def test_empty_action_tool_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        ActionRequest(
            tool_name="",
            arguments={},
            original_text="texto",
            source="local_text",
        )


def test_policy_engine_does_not_invoke_executor() -> None:
    called = {
        "value": False,
    }

    def fake_executor() -> ToolResult:
        called["value"] = True
        return ToolResult(
            success=True,
            message="called",
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="safe_tool",
            description="Tool segura.",
            risk_level=RiskLevel.LOW,
            required_arguments=(),
            executor=fake_executor,
        )
    )

    decision = PolicyEngine(
        registry=registry,
    ).evaluate(
        ActionRequest(
            tool_name="safe_tool",
            arguments={},
            original_text="haz algo",
            source="local_text",
        )
    )

    assert decision.decision is PolicyDecisionType.ALLOW
    assert called["value"] is False


def _registry_with_tool(
    risk_level: RiskLevel,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="generic_tool",
            description="Tool generica.",
            risk_level=risk_level,
            required_arguments=(),
        )
    )

    return registry


def _generic_action() -> ActionRequest:
    return ActionRequest(
        tool_name="generic_tool",
        arguments={},
        original_text="haz algo",
        source="local_text",
    )
