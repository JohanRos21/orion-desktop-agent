from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from orion.tools.contracts import RiskLevel


class PolicyDecisionType(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    decision: PolicyDecisionType
    risk_level: RiskLevel
    reason: str
    tool_name: str | None
    arguments: dict[str, object]
    evaluation_ms: float
