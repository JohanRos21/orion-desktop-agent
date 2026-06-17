from __future__ import annotations


class ExecutionError(RuntimeError):
    """Error base de la capa de ejecucion controlada."""


class ToolExecutionError(ExecutionError):
    """El executor registrado fallo durante la ejecucion."""
