from __future__ import annotations

from time import perf_counter

from orion.commands import handle_command
from orion.tools.applications import normalize_text
from orion.voice import VoiceResult, listen_once


EXIT_COMMANDS = {
    "salir",
    "cerrar",
    "adios",
}

VOICE_COMMANDS = {
    "voz",
    "escuchar",
    "microfono",
}

TIMING_LABELS = {
"microphone_setup": "Preparar micrófono",
"audio_capture": "Grabar audio",
"save_audio": "Guardar audio",
"model_ready": "Preparar modelo",
"transcription": "Transcripción",
"command_execution": "Ejecutar comando",
"vad_model_ready": "Preparar VAD",
"waiting_for_speech": "Esperar voz",
"speech_capture": "Captura VAD",
"vad_total": "Proceso VAD",
}


def print_voice_timings(
    voice_result: VoiceResult,
) -> None:
    if not voice_result.timings_ms:
        return

    print("ORION [TIEMPOS] >")

    for key, duration_ms in voice_result.timings_ms.items():
        label = TIMING_LABELS.get(
            key,
            key,
        )

        if duration_ms >= 1000:
            duration_text = (
                f"{duration_ms / 1000:.2f} s"
            )
        else:
            duration_text = (
                f"{duration_ms:.2f} ms"
            )

        print(
            f"  {label:<18} {duration_text}"
        )


def main() -> None:
    print("=" * 64)
    print("O.R.I.O.N. Desktop Agent")
    print(
        "Operational Reasoning, Interaction "
        "and Orchestration Network"
    )
    print(
        "Versión 0.1.0 — "
        "Experimento de voz local"
    )
    print("Escribe «voz» para hablar.")
    print("Escribe «salir» para cerrar.")
    print("=" * 64)

    while True:
        try:
            command = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(
                "\nORION > Cerrando el sistema."
            )
            break

        normalized_command = normalize_text(
            command
        )

        voice_result: VoiceResult | None = None

        if normalized_command in VOICE_COMMANDS:
            voice_result = listen_once()

            if not voice_result.success:
                print(
                    "ORION [ERROR] > "
                    f"{voice_result.message}"
                )
                print_voice_timings(
                    voice_result
                )
                continue

            command = (
                voice_result.transcript or ""
            )

            normalized_command = normalize_text(
                command
            )

            print(
                f"Tú [VOZ] > {command}"
            )

        if normalized_command in EXIT_COMMANDS:
            print("ORION > Hasta luego.")
            break

        execution_started_at = perf_counter()
        result = handle_command(command)

        execution_ms = round(
            (
                perf_counter()
                - execution_started_at
            )
            * 1000,
            2,
        )

        status = (
            "OK"
            if result.success
            else "ERROR"
        )

        print(
            f"ORION [{status}] > "
            f"{result.message}"
        )

        if voice_result is not None:
            voice_result.timings_ms[
                "command_execution"
            ] = execution_ms

            print_voice_timings(
                voice_result
            )


if __name__ == "__main__":
    main()

