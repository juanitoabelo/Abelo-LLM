"""Sandboxed Python code execution tool."""

from __future__ import annotations

import sys
import traceback
from io import StringIO

from src.tools.registry import ToolResult


_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr,
    "dict": dict, "dir": dir, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format, "frozenset": frozenset,
    "hash": hash, "hex": hex, "id": id, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next, "object": object,
    "oct": oct, "ord": ord, "pow": pow, "range": range, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    "print": print,  # allowed but output is captured
}

_BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket", "ctypes",
    "signal", "multiprocessing", "threading", "importlib",
    "pickle", "shelve", "marshal", "code", "codeop",
    "compile", "exec", "eval", "open", "__builtins__",
}


def code_execute(code: str, timeout: int = 10) -> ToolResult:
    if not code.strip():
        return ToolResult(False, "", "No code provided")

    if len(code) > 5000:
        return ToolResult(False, "", "Code too long (max 5000 chars)")

    combined = ""
    for kw in _BLOCKED_IMPORTS:
        if f"import {kw}" in code or f"from {kw}" in code:
            combined += f"Blocked import: {kw}; "

    if combined:
        return ToolResult(False, "", combined.strip("; "))

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_out = StringIO()
    captured_err = StringIO()
    sys.stdout = captured_out
    sys.stderr = captured_err

    try:
        namespace = {"__builtins__": _SAFE_BUILTINS}
        exec(code, namespace)
        output = captured_out.getvalue()
        error = captured_err.getvalue()
        result = output if output else "(no output)"
        if error:
            result += f"\n--- stderr ---\n{error}"
        return ToolResult(True, result.strip())
    except Exception as e:
        tb = traceback.format_exc()
        return ToolResult(False, "", f"{type(e).__name__}: {e}\n{tb[:500]}")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
