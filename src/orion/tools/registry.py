from __future__ import annotations

from orion.tools.contracts import ToolDefinition


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        tool_definition: ToolDefinition,
    ) -> None:
        tool_name = _normalize_tool_name(
            tool_definition.name
        )

        if tool_name in self._tools:
            raise ValueError(
                f"La herramienta {tool_name!r} ya esta registrada."
            )

        if tool_name != tool_definition.name:
            tool_definition = ToolDefinition(
                name=tool_name,
                description=tool_definition.description,
                risk_level=tool_definition.risk_level,
                required_arguments=(
                    tool_definition.required_arguments
                ),
                executor=tool_definition.executor,
            )

        self._tools[tool_name] = tool_definition

    def get(
        self,
        tool_name: str,
    ) -> ToolDefinition | None:
        return self._tools.get(
            _normalize_tool_name(tool_name)
        )

    def contains(
        self,
        tool_name: str,
    ) -> bool:
        return self.get(tool_name) is not None

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            self._tools.values()
        )


def _normalize_tool_name(
    tool_name: str,
) -> str:
    normalized_name = tool_name.strip()

    if not normalized_name:
        raise ValueError(
            "El nombre de la herramienta no puede estar vacio."
        )

    return normalized_name
