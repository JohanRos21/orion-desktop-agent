from __future__ import annotations


class WakeWordError(RuntimeError):
    """Base error for the isolated wake word subsystem."""


class WakeWordDependencyError(WakeWordError):
    """A required optional dependency is not installed."""


class WakeWordModelError(WakeWordError):
    """The wake word model cannot be loaded or validated."""


class WakeWordAudioError(WakeWordError):
    """The microphone stream cannot be opened or read."""


class UnsupportedWakeWordModelError(WakeWordModelError):
    """The model format is not supported by this platform."""
