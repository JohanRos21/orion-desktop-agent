from orion.tools.applications import (
    ApplicationValidationError,
    validate_application_request,
)

import pytest


@pytest.mark.parametrize(
    ("alias", "canonical_name", "executable"),
    [
        ("calculadora", "calculadora", "calc.exe"),
        ("calculator", "calculadora", "calc.exe"),
        ("calc", "calculadora", "calc.exe"),
        ("bloc de notas", "bloc de notas", "notepad.exe"),
        ("notepad", "bloc de notas", "notepad.exe"),
        ("explorador", "explorador", "explorer.exe"),
        ("explorer", "explorador", "explorer.exe"),
        (
            "explorador de archivos",
            "explorador",
            "explorer.exe",
        ),
    ],
)
def test_allowed_application_aliases_resolve(
    alias: str,
    canonical_name: str,
    executable: str,
) -> None:
    application = validate_application_request(alias)

    assert application.canonical_name == canonical_name
    assert application.executable == executable


@pytest.mark.parametrize(
    "application_name",
    [
        "powershell",
        "terminal",
        "cmd",
        "cmd.exe",
        "simbolo del sistema",
        "regedit",
        "registro",
        "administrador de tareas",
        "configuracion",
        "C:\\Temp\\tool.exe",
        "calculadora && powershell",
    ],
)
def test_blocked_application_names_are_rejected(
    application_name: str,
) -> None:
    with pytest.raises(ApplicationValidationError):
        validate_application_request(application_name)
