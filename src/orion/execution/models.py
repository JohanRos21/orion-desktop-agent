from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionAuditRecord:
    execution_requested: bool
    execution_enabled: bool
    execution_attempted: bool
    execution_success: bool
    execution_ms: float
    result_message: str
