"""Static semantic checking for FTQP programs."""

from ftqp_prototype.typechecker.checker import CheckResult, TypeChecker
from ftqp_prototype.typechecker.errors import ValidationError

__all__ = ["CheckResult", "TypeChecker", "ValidationError"]
