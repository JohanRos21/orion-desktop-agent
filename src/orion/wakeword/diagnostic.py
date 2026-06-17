from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from orion.config import settings
from orion.wakeword.audio_stream import WakeWordAudioConfig
from orion.wakeword.exceptions import WakeWordError
from orion.wakeword.service import (
    WakeWordService,
    describe_microphone,
    prepare_wakeword_resources,
)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostico aislado de palabra de activacion local."
        )
    )
    parser.add_argument(
        "--model-path",
        default=settings.WAKEWORD_MODEL_PATH,
        help="Ruta explicita a un modelo ONNX.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.WAKEWORD_THRESHOLD,
        help="Score minimo para deteccion.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Segundos maximos de escucha.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=settings.WAKEWORD_DEBUG,
        help="Muestra metricas agregadas al finalizar.",
    )
    parser.add_argument(
        "--prepare-models",
        action="store_true",
        help=(
            "Operacion explicita para descargar/localizar recursos "
            "preentrenados de openWakeWord."
        ),
    )
    args = parser.parse_args(
        argv,
    )

    try:
        if args.prepare_models:
            resources_path = prepare_wakeword_resources(
                download=True,
            )
            print(
                "ORION [WAKEWORD] > Recursos en: "
                f"{resources_path}"
            )
            return 0

        audio_config = WakeWordAudioConfig()
        microphone_name, sample_rate = describe_microphone(
            audio_config,
        )
        service = WakeWordService(
            audio_config=audio_config,
            threshold=args.threshold,
            model_path=(
                Path(args.model_path)
                if args.model_path
                else None
            ),
        )

        print(
            "ORION [WAKEWORD] > Microfono: "
            f"{microphone_name}, {sample_rate} Hz"
        )
        print(
            "ORION [WAKEWORD] > Modelo: "
            f"{service.model_name}"
        )
        print(
            "ORION [WAKEWORD] > Esperando palabra de activacion..."
        )

        started_at = perf_counter()
        event = service.listen(
            duration_seconds=args.duration,
        )
        total_ms = (
            perf_counter()
            - started_at
        ) * 1000
        stats = service.debug_stats(
            total_ms=total_ms,
        )

    except WakeWordError as error:
        print(
            "ORION [WAKEWORD] > Error: "
            f"{error}"
        )
        return 1

    if event is not None:
        print(
            "ORION [WAKEWORD] > Detectado: "
            f"{event.wake_word}"
        )
        print(f"Score: {event.score:.2f}")
    else:
        print(
            "ORION [WAKEWORD] > Tiempo agotado sin deteccion."
        )

    if args.debug:
        _print_debug(
            stats,
        )

    return 0


def _print_debug(
    stats,
) -> None:
    print(
        "sample rate nativo: "
        f"{stats.native_sample_rate}"
    )
    print(
        "sample rate del modelo: "
        f"{stats.model_sample_rate}"
    )
    print(
        "frames nativos recibidos: "
        f"{stats.native_frames_received}"
    )
    print(
        "frames a 16 kHz producidos: "
        f"{stats.model_frames_produced}"
    )
    print(
        "bloques de inferencia: "
        f"{stats.inference_blocks}"
    )
    print(
        "score maximo: "
        f"{stats.max_score}"
    )
    print(
        "score promedio: "
        f"{stats.average_score}"
    )
    print(f"overflows: {stats.overflows}")
    print(f"status: {stats.status_count}")
    print(f"tiempo total: {stats.total_ms:.2f} ms")


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
