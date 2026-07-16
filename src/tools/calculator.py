from __future__ import annotations

import math
import re

from src.tools.registry import ToolResult


def calculator(expression: str) -> ToolResult:
    sanitized = _sanitize(expression)
    if sanitized is None:
        return ToolResult(False, "", f"Unsafe expression: {expression}")
    try:
        result = _safe_eval(sanitized)
        return ToolResult(True, str(result))
    except Exception as e:
        return ToolResult(False, "", f"Calculation error: {e}")


_ALLOWED = re.compile(r"^[\d\s+\-*/().,%^!@a-zA-Z_]+$")


def _sanitize(expr: str) -> str | None:
    cleaned = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    if not _ALLOWED.match(cleaned):
        return None
    if any(kw in cleaned for kw in ["__", "import", "exec", "eval", "open", "os.", "sys.", "subprocess"]):
        return None
    return cleaned


_SAFE_ENV: dict = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "int": int, "float": float, "bool": bool, "str": str,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "pow": pow, "pi": math.pi, "e": math.e,
    "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
    "degrees": math.degrees, "radians": math.radians,
}


def _safe_eval(expr: str) -> float:
    code = compile(expr, "<calculator>", "eval")
    for var in code.co_names:
        if var not in _SAFE_ENV:
            raise NameError(f"Name '{var}' is not allowed")
    return eval(code, {"__builtins__": {}}, _SAFE_ENV)
