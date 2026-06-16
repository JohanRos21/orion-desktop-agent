from __future__ import annotations

from orion.llm.exceptions import OllamaError
from orion.llm.intent_parser import interpret_intent


EXIT_COMMANDS = {
    "salir",
    "exit",
    "quit",
}


def _print_result(
    original_input: str,
) -> None:
    result = interpret_intent(
        original_input,
    )
    interpretation = result.interpretation

    print("ORION [LLM] >")
    print(
        f"  Texto original          {interpretation.original_text}"
    )
    print(
        f"  Texto normalizado      {interpretation.normalized_text}"
    )
    print(
        f"  Intent                 {interpretation.intent.value}"
    )
    print(
        "  Aplicacion             "
        f"{interpretation.application_name or ''}"
    )
    print(
        "  Necesita aclaracion    "
        f"{interpretation.needs_clarification}"
    )
    print(
        "  Pregunta de aclaracion "
        f"{interpretation.clarification_question or ''}"
    )
    print(
        "  Respuesta conversacional "
        f"{interpretation.assistant_reply or ''}"
    )
    print(
        f"  Tiempo de Ollama       {result.duration_ms:.2f} ms"
    )


def main() -> None:
    print("ORION LLM Demo")
    print("Escribe texto para interpretar. Usa salir, exit o quit.")

    while True:
        try:
            text = input("\nTexto > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nORION [LLM] > Cerrando demo.")
            break

        if not text:
            continue

        if text.casefold() in EXIT_COMMANDS:
            print("ORION [LLM] > Cerrando demo.")
            break

        try:
            _print_result(text)
        except OllamaError as error:
            print(
                "ORION [LLM ERROR] > "
                f"{error}"
            )


if __name__ == "__main__":
    main()
