from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Resultado estandar producido por una herramienta de ORION."""

    success: bool
    tool_name: str = ""
    message: str = ""
    data: dict[str, object] = field(default_factory=dict)
    duration_ms: float = 0.0
