from orion.models import ToolResult
from orion.tools.applications import normalize_text, open_application


def handle_command(command: str) -> ToolResult:
    """Interpreta los comandos de texto disponibles en ORION V0.1."""
    clean_command = command.strip()
    normalized_command = normalize_text(clean_command)

    if not normalized_command:
        return ToolResult(
            success=False,
            message="No escribiste ningún comando.",
        )

    for prefix in ("abre ", "abrir ", "abri "):
        if normalized_command.startswith(prefix):
            application_name = clean_command[len(prefix):].strip()

            if not application_name:
                return ToolResult(
                    success=False,
                    message="Indica qué aplicación quieres abrir.",
                )

            return open_application(application_name)

    return ToolResult(
        success=False,
        message="Todavía no reconozco ese comando.",
    )
