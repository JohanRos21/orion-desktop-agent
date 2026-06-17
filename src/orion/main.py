from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from orion.llm.exceptions import OllamaConfigurationError
from orion.llm.models import IntentType
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
InputFunction = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class SessionMetrics:
    processed_turns: int
    session_elapsed_ms: float
    processing_total_ms: float
    processing_average_ms: float


def main(
    argv: Sequence[str] | None = None,
    pipeline_factory: PipelineFactory = VoicePipeline,
    input_func: InputFunction = input,
) -> int:
    parser = build_parser()
    args = parser.parse_args(
        argv,
    )

    execution_enabled = bool(
        args.execute
    )

    if args.session:
        return run_session(
            execution_enabled=execution_enabled,
            pipeline_factory=pipeline_factory,
            input_func=input_func,
        )

    return run_single_turn(
        execution_enabled=execution_enabled,
        pipeline_factory=pipeline_factory,
    )


def run_single_turn(
    execution_enabled: bool,
    pipeline_factory: PipelineFactory = VoicePipeline,
) -> int:
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
    except KeyboardInterrupt:
        print("ORION > Sesion finalizada.")
        return EXIT_OK
    except (OllamaConfigurationError, ValueError) as error:
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


def run_session(
    execution_enabled: bool,
    pipeline_factory: PipelineFactory = VoicePipeline,
    input_func: InputFunction = input,
) -> int:
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
    print("Sesion continua iniciada.")
    print()
    print("Presiona Enter para hablar.")
    print("Ctrl+C para cerrar.")

    try:
        pipeline = pipeline_factory()
    except (OllamaConfigurationError, ValueError) as error:
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

    session_id = str(
        uuid4()
    )
    session_started_at = perf_counter()
    turn_count = 0
    processing_total_ms = 0.0

    while True:
        try:
            input_func("")
        except (EOFError, KeyboardInterrupt):
            _print_session_finished(
                turn_count=turn_count,
                session_started_at=session_started_at,
                processing_total_ms=processing_total_ms,
            )
            return EXIT_OK

        turn_number = turn_count + 1
        print()
        print(f"Turno: {turn_number}")

        try:
            result = pipeline.run(
                execution_enabled=execution_enabled,
                session_id=session_id,
                turn_number=turn_number,
            )
        except KeyboardInterrupt:
            _print_session_finished(
                turn_count=turn_count,
                session_started_at=session_started_at,
                processing_total_ms=processing_total_ms,
            )
            return EXIT_OK
        except (OllamaConfigurationError, ValueError) as error:
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

        turn_count += 1
        processing_total_ms += _turn_processing_ms(
            result,
        )

        print_summary(
            result,
            total_label="Tiempo total del turno",
        )

        exit_code = exit_code_for_result(
            result,
        )

        if is_end_session_result(result):
            _print_session_finished(
                turn_count=turn_count,
                session_started_at=session_started_at,
                processing_total_ms=processing_total_ms,
            )
            return EXIT_OK

        if exit_code == EXIT_CONFIGURATION_ERROR:
            _print_session_finished(
                turn_count=turn_count,
                session_started_at=session_started_at,
                processing_total_ms=processing_total_ms,
            )
            return exit_code

        if exit_code == EXIT_APPLICATION_ERROR:
            _print_session_finished(
                turn_count=turn_count,
                session_started_at=session_started_at,
                processing_total_ms=processing_total_ms,
            )
            return exit_code

        print()
        print("Presiona Enter para hablar.")


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
    parser.add_argument(
        "--session",
        action="store_true",
        help="Mantiene ORION activo para procesar varias ordenes.",
    )

    return parser


def print_summary(
    result: VoicePipelineResult,
    total_label: str = "Tiempo total",
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
        f"{total_label}: "
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


def is_end_session_result(
    result: VoicePipelineResult,
) -> bool:
    return (
        result.interpretation is not None
        and result.interpretation.intent is IntentType.END_SESSION
    )


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


def _turn_processing_ms(
    result: VoicePipelineResult,
) -> float:
    return result.timings_ms.get(
        "total",
        0.0,
    )


def _print_session_finished(
    turn_count: int,
    session_started_at: float,
    processing_total_ms: float,
) -> None:
    metrics = _build_session_metrics(
        turn_count=turn_count,
        session_started_at=session_started_at,
        processing_total_ms=processing_total_ms,
    )

    print()
    print("ORION > Sesion finalizada.")
    print(f"Turnos procesados: {metrics.processed_turns}")
    print(
        "Duracion real de la sesion: "
        + _format_ms(metrics.session_elapsed_ms)
    )
    print(
        "Tiempo total procesando: "
        + _format_ms(metrics.processing_total_ms)
    )
    print(
        "Promedio de procesamiento por turno: "
        + _format_ms(metrics.processing_average_ms)
    )


def _build_session_metrics(
    turn_count: int,
    session_started_at: float,
    processing_total_ms: float,
) -> SessionMetrics:
    duration_ms = round(
        (perf_counter() - session_started_at) * 1000,
        2,
    )
    average_processing_ms = (
        round(processing_total_ms / turn_count, 2)
        if turn_count
        else 0.0
    )

    return SessionMetrics(
        processed_turns=turn_count,
        session_elapsed_ms=duration_ms,
        processing_total_ms=round(
            processing_total_ms,
            2,
        ),
        processing_average_ms=average_processing_ms,
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
