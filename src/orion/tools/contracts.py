from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from orion.models import ToolResult


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


ALLOWED_ACTION_SOURCES = {
    "local_text",
    "local_voice",
}


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    risk_level: RiskLevel
    required_arguments: tuple[str, ...]
    executor: Callable[..., ToolResult] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "ToolDefinition.name no puede estar vacio."
            )


@dataclass(frozen=True, slots=True)
class ActionRequest:
    tool_name: str
    arguments: dict[str, object]
    original_text: str
    source: str

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError(
                "ActionRequest.tool_name no puede estar vacio."
            )
