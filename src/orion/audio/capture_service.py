from __future__ import annotations

from orion.audio.models import CapturedAudio
from orion.config import settings


class UnsupportedCaptureModeError(ValueError):
    """Modo de captura de audio no soportado."""


def capture_audio() -> CapturedAudio:
    capture_mode = settings.VOICE_CAPTURE_MODE.strip().casefold()

    if capture_mode == "fixed":
        from orion.audio.fixed_capture import capture_fixed_audio

        return capture_fixed_audio(
            duration_seconds=settings.FIXED_CAPTURE_SECONDS,
        )

    if capture_mode == "vad":
        from orion.audio.vad_capture import capture_vad_audio

        return capture_vad_audio()

    raise UnsupportedCaptureModeError(
        "Modo de captura de voz no soportado: "
        f"{settings.VOICE_CAPTURE_MODE!r}."
    )
