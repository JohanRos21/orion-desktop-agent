import re
import subprocess
import unicodedata

from orion.models import ToolResult


_ALLOWED_APPLICATIONS: dict[str, str] = {
    "bloc de notas": "notepad.exe",
    "notepad": "notepad.exe",
    "explorador": "explorer.exe",
    "explorador de archivos": "explorer.exe",
    "calculadora": "calc.exe",
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


def normalize_text(text: str) -> str:
    """Normaliza mayúsculas, acentos, puntuación y espacios."""
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
    """Obtiene el ejecutable de una aplicación registrada."""
    normalized_name = normalize_text(name)

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

    return _ALLOWED_APPLICATIONS.get(normalized_name)


def open_application(name: str) -> ToolResult:
    """Abre una aplicación permitida por ORION."""
    executable = resolve_application(name)

    if executable is None:
        return ToolResult(
            success=False,
            message=f"No tengo registrada la aplicación «{name}».",
        )

    try:
        subprocess.Popen(
            [executable],
            shell=False,
        )
    except OSError as error:
        return ToolResult(
            success=False,
            message=f"No pude abrir «{name}»: {error}",
        )

    return ToolResult(
        success=True,
        message=f"Abrí «{name}».",
    )
