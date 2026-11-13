"""Fault sources for the source-aware error analysis (arXiv-2608.04573 §5).

Each fault source is an independent event with probability p(s). A codeblock
carries the set of sources that may have disturbed it, and its error rate is
the risk R(U) = 1 - prod(1 - p(s)) of that set.

Risks are kept in log-survival form, log(1 - R) = sum(log1p(-p)), so that
probabilities around 1e-12 keep full precision.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field


def log_survival(p: float) -> float:
    """log(1 - p), with p clamped to [0, 1]."""
    p = min(1.0, max(0.0, p))
    if p >= 1.0:
        return -math.inf
    return math.log1p(-p)


def risk_from_log_survival(logq: float) -> float:
    """1 - exp(logq), i.e. the risk of a set whose log-survival is logq."""
    if logq == -math.inf:
        return 1.0
    return min(1.0, max(0.0, -math.expm1(logq)))


@dataclass
class SourceTable:
    """Global parameters p(s) of one derivation. Ids are never reused."""

    p: list[float] = field(default_factory=list)
    logq: list[float] = field(default_factory=list)
    kind: list[str] = field(default_factory=list)
    label: list[str] = field(default_factory=list)

    def fresh(self, p: float, kind: str, label: str = "") -> int:
        p = min(1.0, max(0.0, p))
        self.p.append(p)
        self.logq.append(log_survival(p))
        self.kind.append(kind)
        self.label.append(label)
        return len(self.p) - 1

    def log_survival(self, ids: Iterable[int]) -> float:
        return math.fsum(self.logq[i] for i in ids)

    def risk(self, ids: Iterable[int]) -> float:
        return risk_from_log_survival(self.log_survival(ids))

    def __len__(self) -> int:
        return len(self.p)
