from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Resultado estándar producido por una herramienta de ORION."""

    success: bool
    message: str