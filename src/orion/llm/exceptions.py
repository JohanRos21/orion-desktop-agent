from __future__ import annotations


class OllamaError(RuntimeError):
    """Error base de la integracion local con Ollama."""


class OllamaConfigurationError(OllamaError):
    """La configuracion de Ollama no es segura o valida."""


class OllamaConnectionError(OllamaError):
    """No se pudo conectar con el servidor local de Ollama."""


class OllamaTimeoutError(OllamaError):
    """Ollama no respondio dentro del tiempo configurado."""


class OllamaModelNotFoundError(OllamaError):
    """El modelo configurado no existe en Ollama."""


class OllamaInvalidResponseError(OllamaError):
    """Ollama devolvio una respuesta que no cumple el contrato."""
