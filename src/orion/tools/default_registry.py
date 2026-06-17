from __future__ import annotations

from orion.tools.applications import open_application
from orion.tools.contracts import (
    RiskLevel,
    ToolDefinition,
)
from orion.tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="open_application",
            description="Abre una aplicacion permitida de Windows.",
            risk_level=RiskLevel.LOW,
            required_arguments=("application_name",),
            executor=open_application,
        )
    )

    return registry
