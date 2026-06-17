from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

from orion.config import settings
from orion.wakeword.audio_stream import (
    WakeWordAudioConfig,
    WakeWordAudioStream,
    WakeWordFrameProcessor,
)
from orion.wakeword.exceptions import (
    UnsupportedWakeWordModelError,
    WakeWordDependencyError,
    WakeWordModelError,
)
from orion.wakeword.models import (
    WakeWordDebugStats,
    WakeWordEvent,
    WakeWordPrediction,
)


DEFAULT_WAKEWORD_NAME = "hey jarvis"
DEFAULT_CUSTOM_MODEL_PATH = Path(
    "local_models/wakeword/orion.onnx"
)


@dataclass(frozen=True, slots=True)
class WakeWordModelReference:
    model_ref: str
    display_name: str
    path: Path | None = None


class OpenWakeWordModelAdapter:
    def __init__(
        self,
        model_path: str | Path | None = None,
        model_name: str = DEFAULT_WAKEWORD_NAME,
        model_loader: Any | None = None,
    ) -> None:
        self.reference = resolve_model_reference(
            model_path=model_path,
            model_name=model_name,
        )
        self._model_loader = model_loader
        self._model: Any | None = None

    @property
    def display_name(
        self,
    ) -> str:
        return self.reference.display_name

    def load(
        self,
    ) -> Any:
        if self._model is not None:
            return self._model

        if self._model_loader is not None:
            self._model = self._model_loader(
                self.reference,
            )
            return self._model

        try:
            from openwakeword.model import Model
        except ImportError as error:
            raise WakeWordDependencyError(
                "Falta la dependencia 'openwakeword'. Instala las "
                "dependencias antes de iniciar el detector."
            ) from error

        try:
            self._model = Model(
                wakeword_models=[
                    self.reference.model_ref,
                ],
                inference_framework="onnx",
                vad_threshold=0,
            )
        except Exception as error:
            raise WakeWordModelError(
                "No pude cargar el modelo wake word ONNX. "
                "Ejecuta el diagnostico/preparacion de recursos o "
                "proporciona --model-path a un .onnx existente. "
                f"Detalle: {error}"
            ) from error

        return self._model

    def predict(
        self,
        frame_pcm16: Any,
    ) -> dict[str, float]:
        model = self.load()
        raw_predictions = model.predict(
            frame_pcm16,
        )

        return {
            str(model_name): float(score)
            for model_name, score in raw_predictions.items()
        }


class WakeWordService:
    def __init__(
        self,
        audio_config: WakeWordAudioConfig | None = None,
        threshold: float = settings.WAKEWORD_THRESHOLD,
        cooldown_seconds: float = settings.WAKEWORD_COOLDOWN_SECONDS,
        model_path: str | Path | None = settings.WAKEWORD_MODEL_PATH,
        model_adapter: OpenWakeWordModelAdapter | None = None,
        audio_stream: WakeWordAudioStream | None = None,
        frame_processor: WakeWordFrameProcessor | None = None,
        time_func: Any = monotonic,
    ) -> None:
        self.audio_config = audio_config or WakeWordAudioConfig()
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.model_adapter = model_adapter or OpenWakeWordModelAdapter(
            model_path=model_path,
        )
        self.audio_stream = audio_stream or WakeWordAudioStream(
            config=self.audio_config,
        )
        self.frame_processor = frame_processor or WakeWordFrameProcessor(
            config=self.audio_config,
        )
        self.time_func = time_func
        self._last_detection_at: float | None = None
        self._started = False
        self._score_sum = 0.0
        self._score_count = 0
        self._max_score = 0.0

    @property
    def model_name(
        self,
    ) -> str:
        return self.model_adapter.display_name

    def start(
        self,
    ) -> None:
        self.model_adapter.load()
        self.audio_stream.start()
        self._started = True

    def stop(
        self,
    ) -> None:
        self.audio_stream.stop()
        self._started = False

    def process_audio(
        self,
        native_block: Any,
    ) -> list[WakeWordEvent]:
        events: list[WakeWordEvent] = []

        for frame in self.frame_processor.process_native_block(
            native_block,
        ):
            prediction = self._predict_frame(
                frame,
            )

            if not prediction.detected:
                continue

            if not self._cooldown_elapsed(
                prediction.timestamp,
            ):
                continue

            self._last_detection_at = prediction.timestamp
            events.append(
                WakeWordEvent(
                    wake_word=prediction.model_name,
                    score=prediction.score,
                    detected_at=prediction.timestamp,
                )
            )

        return events

    def listen(
        self,
        duration_seconds: float,
        read_timeout_seconds: float = 0.1,
    ) -> WakeWordEvent | None:
        started_here = not self._started

        if started_here:
            self.start()

        deadline = self.time_func() + duration_seconds

        try:
            while self.time_func() < deadline:
                block = self.audio_stream.read(
                    timeout=read_timeout_seconds,
                )
                if block is None:
                    continue

                events = self.process_audio(
                    block,
                )
                if events:
                    return events[0]
        finally:
            if started_here:
                self.stop()

        return None

    def debug_stats(
        self,
        total_ms: float,
    ) -> WakeWordDebugStats:
        average_score = (
            self._score_sum / self._score_count
            if self._score_count
            else 0.0
        )

        return WakeWordDebugStats(
            native_sample_rate=self.audio_config.native_sample_rate,
            model_sample_rate=self.audio_config.model_sample_rate,
            native_frames_received=self.frame_processor.native_frames_received,
            model_frames_produced=self.frame_processor.model_frames_produced,
            inference_blocks=self.frame_processor.inference_blocks,
            max_score=round(
                self._max_score,
                6,
            ),
            average_score=round(
                average_score,
                6,
            ),
            overflows=self.audio_stream.overflow_count,
            status_count=self.audio_stream.status_count,
            total_ms=round(
                total_ms,
                2,
            ),
        )

    def _predict_frame(
        self,
        frame: Any,
    ) -> WakeWordPrediction:
        scores = self.model_adapter.predict(
            frame,
        )
        model_name, score = _max_score(
            scores,
            fallback_model_name=self.model_name,
        )
        self._score_sum += score
        self._score_count += 1
        self._max_score = max(
            self._max_score,
            score,
        )

        return WakeWordPrediction(
            model_name=model_name,
            score=score,
            detected=score >= self.threshold,
            timestamp=self.time_func(),
        )

    def _cooldown_elapsed(
        self,
        detected_at: float,
    ) -> bool:
        if self._last_detection_at is None:
            return True

        return (
            detected_at
            - self._last_detection_at
        ) >= self.cooldown_seconds


def resolve_model_reference(
    model_path: str | Path | None,
    model_name: str = DEFAULT_WAKEWORD_NAME,
) -> WakeWordModelReference:
    if model_path is None:
        return WakeWordModelReference(
            model_ref=model_name,
            display_name=model_name,
        )

    path = Path(
        model_path,
    )
    suffix = path.suffix.casefold()

    if suffix == ".tflite":
        raise UnsupportedWakeWordModelError(
            "Los modelos TFLite estan rechazados en Windows; usa ONNX."
        )

    if suffix != ".onnx":
        raise WakeWordModelError(
            "El modelo wake word debe ser un archivo .onnx."
        )

    if not path.exists():
        raise WakeWordModelError(
            f"No existe el modelo wake word: {path}."
        )

    return WakeWordModelReference(
        model_ref=str(path),
        display_name=path.stem,
        path=path,
    )


def describe_microphone(
    audio_config: WakeWordAudioConfig | None = None,
) -> tuple[str, int]:
    config = audio_config or WakeWordAudioConfig()

    try:
        import sounddevice as sd
    except ImportError as error:
        raise WakeWordDependencyError(
            "Falta la dependencia 'sounddevice'."
        ) from error

    info = sd.query_devices(
        config.device_index,
        kind="input",
    )

    return (
        str(
            info.get(
                "name",
                "microfono",
            )
        ),
        config.native_sample_rate,
    )


def prepare_wakeword_resources(
    download: bool = False,
) -> Path:
    try:
        import openwakeword
    except ImportError as error:
        raise WakeWordDependencyError(
            "Falta la dependencia 'openwakeword'."
        ) from error

    package_path = Path(
        openwakeword.__file__,
    ).resolve().parent

    if download:
        _download_default_onnx_resources(
            openwakeword_module=openwakeword,
            target_directory=(
                package_path
                / "resources"
                / "models"
            ),
        )

    return package_path


def _download_default_onnx_resources(
    openwakeword_module: Any,
    target_directory: Path,
    model_name_fragment: str = "hey_jarvis",
) -> None:
    try:
        import requests
    except ImportError as error:
        raise WakeWordDependencyError(
            "Falta la dependencia 'requests' para descargar recursos."
        ) from error

    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    urls: list[str] = []

    for feature_model in openwakeword_module.FEATURE_MODELS.values():
        urls.append(
            feature_model["download_url"].replace(
                ".tflite",
                ".onnx",
            )
        )

    for model in openwakeword_module.MODELS.values():
        url = model["download_url"]
        filename = url.split("/")[-1]
        if model_name_fragment in filename:
            urls.append(
                url.replace(
                    ".tflite",
                    ".onnx",
                )
            )

    if not urls:
        raise WakeWordModelError(
            "No encontre el modelo ONNX preentrenado hey jarvis."
        )

    for url in urls:
        output_path = target_directory / url.split("/")[-1]
        if output_path.exists():
            continue

        try:
            response = requests.get(
                url,
                timeout=30,
            )
            response.raise_for_status()
        except Exception as error:
            raise WakeWordModelError(
                "No pude descargar el recurso ONNX de openWakeWord: "
                f"{url}. Detalle: {error}"
            ) from error

        output_path.write_bytes(
            response.content,
        )


def _max_score(
    scores: dict[str, float],
    fallback_model_name: str,
) -> tuple[str, float]:
    if not scores:
        return fallback_model_name, 0.0

    return max(
        scores.items(),
        key=lambda item: item[1],
    )


def listen_for_wakeword(
    duration_seconds: float,
    threshold: float = settings.WAKEWORD_THRESHOLD,
    model_path: str | Path | None = settings.WAKEWORD_MODEL_PATH,
) -> tuple[WakeWordEvent | None, WakeWordDebugStats]:
    service = WakeWordService(
        threshold=threshold,
        model_path=model_path,
    )
    started_at = perf_counter()
    event = service.listen(
        duration_seconds=duration_seconds,
    )
    total_ms = (
        perf_counter()
        - started_at
    ) * 1000

    return event, service.debug_stats(
        total_ms=total_ms,
    )
