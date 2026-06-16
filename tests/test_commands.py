from orion.commands import handle_command
from orion.tools.applications import normalize_text, resolve_application


def test_normalize_text_removes_accents() -> None:
    assert normalize_text("  ADIÓS  ") == "adios"


def test_resolve_known_application() -> None:
    assert resolve_application("Bloc de notas") == "notepad.exe"


def test_resolve_application_ignores_case() -> None:
    assert resolve_application("CALCULADORA") == "calc.exe"


def test_resolve_unknown_application() -> None:
    assert resolve_application("aplicación inventada") is None


def test_empty_command_is_rejected() -> None:
    result = handle_command("")

    assert result.success is False
    assert result.message == "No escribiste ningún comando."


def test_missing_application_name_is_rejected() -> None:
    result = handle_command("abre")

    assert result.success is False


def test_unknown_command_is_rejected() -> None:
    result = handle_command("haz algo extraño")

    assert result.success is False
    assert result.message == "Todavía no reconozco ese comando."

def test_resolve_application_accepts_article_and_punctuation() -> None:
    assert resolve_application("La calculadora.") == "calc.exe"


def test_resolve_application_accepts_transcription_filler() -> None:
    assert resolve_application("en la calculadora") == "calc.exe"
