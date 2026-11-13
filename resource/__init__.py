"""Resource estimation models and semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ftqp_prototype.resource.estimator import (
        ResourceEstimate,
        ResourceEstimationError,
        ResourceEstimator,
    )
    from ftqp_prototype.resource.estimator_v2 import (
        ResourceEstimateV2,
        ResourceEstimatorV2,
    )

__all__ = [
    "ResourceEstimate",
    "ResourceEstimationError",
    "ResourceEstimator",
    "ResourceEstimateV2",
    "ResourceEstimatorV2",
]


def __getattr__(name: str) -> object:
    if name in ("ResourceEstimate", "ResourceEstimationError", "ResourceEstimator"):
        from ftqp_prototype.resource import estimator

        return getattr(estimator, name)
    if name in ("ResourceEstimateV2", "ResourceEstimatorV2"):
        from ftqp_prototype.resource import estimator_v2

        return getattr(estimator_v2, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
