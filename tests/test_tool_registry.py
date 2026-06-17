import pytest

from orion.models import ToolResult
from orion.tools.contracts import (
    RiskLevel,
    ToolDefinition,
)
from orion.tools.default_registry import build_default_registry
from orion.tools.registry import ToolRegistry


def test_registry_registers_and_finds_tool() -> None:
    registry = ToolRegistry()
    tool_definition = ToolDefinition(
        name="test_tool",
        description="Tool de prueba.",
        risk_level=RiskLevel.LOW,
        required_arguments=("value",),
        executor=lambda value: ToolResult(
            success=True,
            message=str(value),
        ),
    )

    registry.register(
        tool_definition
    )

    assert registry.contains("test_tool") is True
    assert registry.get("test_tool") == tool_definition


def test_registry_rejects_duplicate_tool() -> None:
    registry = ToolRegistry()
    tool_definition = ToolDefinition(
        name="test_tool",
        description="Tool de prueba.",
        risk_level=RiskLevel.LOW,
        required_arguments=(),
    )

    registry.register(
        tool_definition
    )

    with pytest.raises(ValueError):
        registry.register(
            tool_definition
        )


def test_registry_lists_tools() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="test_tool",
            description="Tool de prueba.",
            risk_level=RiskLevel.LOW,
            required_arguments=(),
        )
    )

    tools = registry.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "test_tool"


def test_registry_rejects_empty_tool_name() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError):
        registry.register(
            ToolDefinition(
                name="",
                description="Tool invalida.",
                risk_level=RiskLevel.LOW,
                required_arguments=(),
            )
        )


def test_default_registry_contains_only_open_application() -> None:
    registry = build_default_registry()
    tools = registry.list_tools()

    assert len(tools) == 1
    assert registry.contains("open_application") is True
    assert tools[0].risk_level is RiskLevel.LOW
    assert tools[0].required_arguments == ("application_name",)
    assert tools[0].executor is not None
