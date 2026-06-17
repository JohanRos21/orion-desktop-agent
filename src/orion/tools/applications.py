from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from time import perf_counter
import unicodedata

from orion.models import ToolResult


@dataclass(frozen=True, slots=True)
class ApplicationDefinition:
    canonical_name: str
    display_name: str
    executable: str


class ApplicationValidationError(ValueError):
    """La aplicacion solicitada no pertenece al catalogo seguro."""


_CALCULATOR = ApplicationDefinition(
    canonical_name="calculadora",
    display_name="Calculadora",
    executable="calc.exe",
)
_NOTEPAD = ApplicationDefinition(
    canonical_name="bloc de notas",
    display_name="Bloc de notas",
    executable="notepad.exe",
)
_EXPLORER = ApplicationDefinition(
    canonical_name="explorador",
    display_name="Explorador",
    executable="explorer.exe",
)

_APPLICATION_ALIASES: dict[str, ApplicationDefinition] = {
    "calculadora": _CALCULATOR,
    "calculator": _CALCULATOR,
    "calc": _CALCULATOR,
    "bloc de notas": _NOTEPAD,
    "notepad": _NOTEPAD,
    "explorador": _EXPLORER,
    "explorer": _EXPLORER,
    "explorador de archivos": _EXPLORER,
}

_BLOCKED_APPLICATION_NAMES = {
    "powershell",
    "terminal",
    "cmd",
    "cmd exe",
    "simbolo del sistema",
    "regedit",
    "registro",
    "administrador de tareas",
    "configuracion",
}

_LEADING_ARTICLES = (
    "en ",
    "a ",
    "el ",
    "la ",
    "los ",
    "las ",
    "un ",
    "una ",
)

_SHELL_PATTERNS = (
    "&&",
    "||",
    ";",
    "|",
    ">",
    "<",
    "powershell",
    "cmd.exe",
)

_PATH_PATTERNS = (
    "\\",
    "/",
    ":",
)


def normalize_text(text: str) -> str:
    """Normaliza mayusculas, acentos, puntuacion y espacios."""
    decomposed_text = unicodedata.normalize(
        "NFD",
        text.casefold().strip(),
    )

    without_accents = "".join(
        character
        for character in decomposed_text
        if unicodedata.category(character) != "Mn"
    )

    without_punctuation = re.sub(
        r"[^\w\s]",
        "",
        without_accents,
    )

    return " ".join(without_punctuation.split())


def resolve_application(name: str) -> str | None:
    """Obtiene el ejecutable seguro de una aplicacion registrada."""
    application = resolve_application_definition(name)

    if application is None:
        return None

    return application.executable


def resolve_application_definition(
    name: str,
) -> ApplicationDefinition | None:
    normalized_name = _strip_leading_articles(
        normalize_text(name)
    )

    return _APPLICATION_ALIASES.get(
        normalized_name
    )


def validate_application_request(
    application_name: object,
    original_text: str = "",
) -> ApplicationDefinition:
    if not isinstance(application_name, str):
        raise ApplicationValidationError(
            "application_name debe ser texto"
        )

    if _looks_like_shell_or_path(application_name):
        raise ApplicationValidationError(
            "application_name parece contener comandos de shell"
        )

    if original_text and _looks_like_shell_or_path(original_text):
        raise ApplicationValidationError(
            "texto original parece contener comandos de shell"
        )

    normalized_name = _strip_leading_articles(
        normalize_text(application_name)
    )

    if not normalized_name:
        raise ApplicationValidationError(
            "application_name no puede estar vacio"
        )

    if normalized_name in _BLOCKED_APPLICATION_NAMES:
        raise ApplicationValidationError(
            "application_name esta bloqueada por politica"
        )

    application = _APPLICATION_ALIASES.get(
        normalized_name
    )

    if application is None:
        raise ApplicationValidationError(
            "application_name no esta en el catalogo seguro"
        )

    return application


def open_application(application_name: str) -> ToolResult:
    """Abre una aplicacion permitida por ORION."""
    started_at = perf_counter()

    try:
        application = validate_application_request(application_name)
    except ApplicationValidationError as error:
        return ToolResult(
            success=False,
            tool_name="open_application",
            message=str(error),
            data={
                "application_name": application_name,
            },
            duration_ms=_milliseconds(started_at),
        )

    try:
        subprocess.Popen(
            [application.executable],
            shell=False,
        )
    except OSError as error:
        return ToolResult(
            success=False,
            tool_name="open_application",
            message=(
                "No pude abrir "
                f"{application.display_name}: {error}"
            ),
            data={
                "application_name": application.canonical_name,
                "executable": application.executable,
            },
            duration_ms=_milliseconds(started_at),
        )

    return ToolResult(
        success=True,
        tool_name="open_application",
        message=f"Abri {application.display_name}.",
        data={
            "application_name": application.canonical_name,
            "executable": application.executable,
        },
        duration_ms=_milliseconds(started_at),
    )


def _strip_leading_articles(
    normalized_name: str,
) -> str:
    removed_prefix = True

    while removed_prefix:
        removed_prefix = False

        for article in _LEADING_ARTICLES:
            if normalized_name.startswith(article):
                normalized_name = (
                    normalized_name[len(article):]
                    .strip()
                )
                removed_prefix = True
                break

    return normalized_name


def _looks_like_shell_or_path(
    text: str,
) -> bool:
    casefolded_text = text.casefold()

    if any(
        pattern in casefolded_text
        for pattern in _SHELL_PATTERNS
    ):
        return True

    if ".exe" in casefolded_text:
        return True

    return any(
        pattern in text
        for pattern in _PATH_PATTERNS
    )


def _milliseconds(
    started_at: float,
) -> float:
    return round(
        (perf_counter() - started_at) * 1000,
        2,
    )
