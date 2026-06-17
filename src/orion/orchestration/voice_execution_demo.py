from __future__ import annotations

import argparse
from collections.abc import Sequence

from orion.orchestration.voice_pipeline import (
    VoicePipeline,
    VoicePipelineResult,
)
from orion.policy.models import PolicyDecisionType
from orion.tools.applications import (
    ApplicationValidationError,
    validate_application_request,
)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description="Demo de voz con policy y ejecucion controlada.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Habilita ejecucion real de herramientas permitidas.",
    )
    args = parser.parse_args(
        argv,
    )

    run_demo(
        execution_enabled=args.execute,
    )


def run_demo(
    execution_enabled: bool,
    pipeline: VoicePipeline | None = None,
) -> VoicePipelineResult:
    active_pipeline = pipeline or VoicePipeline()

    print("ORION Voice Execution Demo")
    print(
        "Ejecucion real habilitada: "
        f"{execution_enabled}"
    )
    print("Habla cuando comience la captura.")

    result = active_pipeline.run(
        execution_enabled=execution_enabled,
    )

    print_result(
        result,
    )

    return result


def print_result(
    result: VoicePipelineResult,
) -> None:
    interpretation = result.interpretation
    action_request = result.action_request
    policy_decision = result.policy_decision

    print("\nTranscripcion")
    print(
        "  original_text = "
        f"{result.transcript or '(vacia)'}"
    )

    print("\nTexto normalizado")
    print(
        "  normalized_text = "
        + (
            interpretation.normalized_text
            if interpretation is not None
            else "(no disponible)"
        )
    )

    print("\nIntent")
    print(
        "  intent = "
        + (
            interpretation.intent.value
            if interpretation is not None
            else "(no disponible)"
        )
    )

    if (
        interpretation is not None
        and interpretation.assistant_reply
    ):
        print("\nRespuesta")
        print(f"  {interpretation.assistant_reply}")

    if (
        interpretation is not None
        and interpretation.clarification_question
    ):
        print("\nAclaracion")
        print(f"  {interpretation.clarification_question}")

    print("\nAplicacion")
    print(
        "  application_name = "
        + _application_name_label(result)
    )
    print(
        "  resolved = "
        + _application_label(result)
    )

    print("\nActionRequest")
    print(
        "  "
        + _action_request_label(
            result,
        )
    )

    print("\nDecision")
    print(
        "  decision = "
        + (
            policy_decision.decision.value.upper()
            if policy_decision is not None
            else "(sin decision)"
        )
    )

    print("\nRiesgo")
    print(
        "  risk = "
        + (
            policy_decision.risk_level.value.upper()
            if policy_decision is not None
            else "(sin riesgo)"
        )
    )

    print("\nEjecucion habilitada")
    print(
        "  execution_enabled = "
        f"{result.execution_enabled}"
    )

    print("\nEjecucion intentada")
    print(
        "  execution_attempted = "
        f"{result.execution_attempted}"
    )

    print("\nResultado")
    print(f"  {result.message}")

    print("\nTiempos por etapa")
    for key in (
        "capture",
        "transcription",
        "llm",
        "routing",
        "policy",
        "execution",
    ):
        print(
            f"  {key:<14} {_format_ms(result.timings_ms.get(key, 0.0))}"
        )

    print("\nTiempo total")
    print(
        "  "
        + _format_ms(
            result.timings_ms.get(
                "total",
                0.0,
            )
        )
    )

    if (
        policy_decision is not None
        and policy_decision.decision is PolicyDecisionType.BLOCK
    ):
        print("\nBloqueo")
        print(f"  {policy_decision.reason}")


def _application_label(
    result: VoicePipelineResult,
) -> str:
    interpretation = result.interpretation

    if (
        interpretation is None
        or not interpretation.application_name
    ):
        return "(no aplica)"

    if result.action_request is None:
        return interpretation.application_name

    try:
        application = validate_application_request(
            application_name=(
                result.action_request.arguments.get(
                    "application_name",
                )
            ),
            original_text=result.action_request.original_text,
        )
    except ApplicationValidationError as error:
        return (
            f"{interpretation.application_name} "
            f"(bloqueada: {error})"
        )

    return (
        f"{application.display_name} "
        f"({application.executable})"
    )


def _application_name_label(
    result: VoicePipelineResult,
) -> str:
    interpretation = result.interpretation

    if (
        interpretation is None
        or not interpretation.application_name
    ):
        return "(no aplica)"

    return interpretation.application_name


def _action_request_label(
    result: VoicePipelineResult,
) -> str:
    action_request = result.action_request

    if action_request is None:
        return "(no creada)"

    return (
        f"tool={action_request.tool_name!r}, "
        f"source={action_request.source!r}, "
        f"arguments={action_request.arguments!r}"
    )


def _format_ms(
    duration_ms: float,
) -> str:
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f} s"

    return f"{duration_ms:.2f} ms"


if __name__ == "__main__":
    main()
