from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from orion.orchestration.voice_pipeline import (
    VoicePipeline,
    VoicePipelineResult,
)
from orion.policy.models import PolicyDecisionType


# Codigos de salida del entrypoint oficial:
# 0 = procesamiento completado correctamente, incluso conversation,
#     unknown, CONFIRM o BLOCK controlado
# 1 = error inesperado de aplicacion
# 2 = error de captura o transcripcion
# 3 = Ollama no disponible, timeout o modelo ausente
# 4 = error interno de configuracion
EXIT_OK = 0
EXIT_APPLICATION_ERROR = 1
EXIT_CAPTURE_OR_TRANSCRIPTION_ERROR = 2
EXIT_OLLAMA_ERROR = 3
EXIT_CONFIGURATION_ERROR = 4


PipelineFactory = Callable[[], VoicePipeline]


def main(
    argv: Sequence[str] | None = None,
    pipeline_factory: PipelineFactory = VoicePipeline,
) -> int:
    parser = build_parser()
    args = parser.parse_args(
        argv,
    )

    execution_enabled = bool(
        args.execute
    )

    print("ORION")
    print()
    print(
        "Modo: "
        + (
            "ejecucion real"
            if execution_enabled
            else "simulacion"
        )
    )
    print("Escuchando...")
    print()

    try:
        pipeline = pipeline_factory()
        result = pipeline.run(
            execution_enabled=execution_enabled,
        )
    except ValueError as error:
        print(
            "Error de configuracion: "
            f"{error}"
        )
        return EXIT_CONFIGURATION_ERROR
    except Exception as error:
        print(
            "Error inesperado de aplicacion: "
            f"{error}"
        )
        return EXIT_APPLICATION_ERROR

    print_summary(
        result,
    )

    return exit_code_for_result(
        result,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ORION escucha una instruccion, la transcribe, "
            "la interpreta con Ollama, evalua politica y "
            "solo ejecuta herramientas permitidas con --execute."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Habilita ejecucion real de herramientas permitidas. "
            "Sin este flag ORION solo simula y no abre aplicaciones."
        ),
    )

    return parser


def print_summary(
    result: VoicePipelineResult,
) -> None:
    interpretation = result.interpretation
    policy_decision = result.policy_decision

    print(
        "Transcripcion: "
        f"{result.transcript or '(vacia)'}"
    )

    if interpretation is not None:
        print(
            "Interpretacion: "
            f"{interpretation.intent.value}"
        )

        if interpretation.application_name:
            print(
                "Aplicacion: "
                f"{interpretation.application_name}"
            )

        if interpretation.assistant_reply:
            print(
                "Respuesta: "
                f"{interpretation.assistant_reply}"
            )

        if interpretation.clarification_question:
            print(
                "Aclaracion: "
                f"{interpretation.clarification_question}"
            )
    else:
        print("Interpretacion: no disponible")

    if policy_decision is not None:
        print(
            "Politica: "
            f"{policy_decision.decision.value.upper()}"
        )

    print(
        "Ejecucion: "
        + _execution_label(
            result,
        )
    )

    print(
        "Resultado: "
        + _result_message(
            result,
        )
    )

    if (
        policy_decision is not None
        and policy_decision.decision is PolicyDecisionType.BLOCK
    ):
        print(
            "Motivo: "
            f"{policy_decision.reason}"
        )

    print(
        "Tiempo total: "
        + _format_ms(
            result.timings_ms.get(
                "total",
                0.0,
            )
        )
    )


def exit_code_for_result(
    result: VoicePipelineResult,
) -> int:
    if result.success:
        return EXIT_OK

    if result.error_stage == "capture_transcription":
        return EXIT_CAPTURE_OR_TRANSCRIPTION_ERROR

    if result.error_stage == "llm":
        return EXIT_OLLAMA_ERROR

    if result.error_stage == "configuration":
        return EXIT_CONFIGURATION_ERROR

    return EXIT_APPLICATION_ERROR


def _execution_label(
    result: VoicePipelineResult,
) -> str:
    if not result.execution_enabled:
        return "deshabilitada"

    if result.execution_attempted:
        return "intentada"

    if result.policy_decision is None:
        return "no requerida"

    if result.policy_decision.decision is PolicyDecisionType.ALLOW:
        return "no intentada"

    return "bloqueada"


def _result_message(
    result: VoicePipelineResult,
) -> str:
    policy_decision = result.policy_decision

    if (
        policy_decision is not None
        and policy_decision.decision is PolicyDecisionType.BLOCK
    ):
        return "Solicitud bloqueada por seguridad."

    if (
        policy_decision is not None
        and policy_decision.decision is PolicyDecisionType.ALLOW
        and not result.execution_enabled
    ):
        return (
            "La accion fue permitida, pero la ejecucion real "
            "esta deshabilitada."
        )

    return result.message


def _format_ms(
    duration_ms: float,
) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f} s"

    return f"{duration_ms:.2f} ms"


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
