"""Safe arithmetic expression evaluator for QEC specifications."""

from __future__ import annotations

import ast as py_ast
import math
import re
from dataclasses import dataclass
from typing import Callable


class ExpressionError(Exception):
    """Raised when a resource expression cannot be evaluated safely."""


_ALLOWED_FUNCS: dict[str, Callable[..., float]] = {
    "max": max,
    "min": min,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
}

_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


@dataclass
class ExpressionEvaluator:
    """Evaluate simple numeric expressions against a variable environment."""

    variables: dict[str, float]

    def eval(self, expr: str | int | float) -> float:
        if isinstance(expr, (int, float)):
            return float(expr)
        normalized = _normalize_expression(expr)
        try:
            tree = py_ast.parse(normalized, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"invalid expression {expr!r}: {exc.msg}") from exc
        return float(self._eval_node(tree.body))

    def _eval_node(self, node: py_ast.AST) -> float:
        if isinstance(node, py_ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ExpressionError(f"unsupported literal {node.value!r}")
        if isinstance(node, py_ast.Name):
            if node.id in self.variables:
                return float(self.variables[node.id])
            if node.id in _ALLOWED_CONSTS:
                return float(_ALLOWED_CONSTS[node.id])
            raise ExpressionError(f"unknown variable {node.id!r}")
        if isinstance(node, py_ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, py_ast.Add):
                return left + right
            if isinstance(node.op, py_ast.Sub):
                return left - right
            if isinstance(node.op, py_ast.Mult):
                return left * right
            if isinstance(node.op, py_ast.Div):
                return left / right
            if isinstance(node.op, py_ast.Pow):
                return left**right
            if isinstance(node.op, py_ast.Mod):
                return left % right
        if isinstance(node, py_ast.UnaryOp):
            value = self._eval_node(node.operand)
            if isinstance(node.op, py_ast.UAdd):
                return value
            if isinstance(node.op, py_ast.USub):
                return -value
        if isinstance(node, py_ast.Call):
            if not isinstance(node.func, py_ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ExpressionError("only whitelisted numeric functions may be called")
            args = [self._eval_node(arg) for arg in node.args]
            return float(_ALLOWED_FUNCS[node.func.id](*args))
        raise ExpressionError(f"unsupported expression form {type(node).__name__}")


def _normalize_expression(expr: str) -> str:
    """Translate the paper's compact notation into Python arithmetic syntax."""

    out = expr.strip()
    out = out.replace("{", "(").replace("}", ")")
    out = out.replace("^", "**")
    # The paper writes examples such as 6t1 and 255tSE. Insert multiplication
    # only where it is unambiguous.
    out = re.sub(r"(?<=[0-9])(?=[A-Za-z_])", "*", out)
    out = re.sub(r"(?<=[A-Za-z_0-9])(?=\()", "*", out)
    # Undo function-name multiplication introduced by the previous rule.
    for func in _ALLOWED_FUNCS:
        out = out.replace(f"{func}*(", f"{func}(")
    return out
