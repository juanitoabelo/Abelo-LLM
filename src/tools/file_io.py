"""Controlled file system I/O tools."""

from __future__ import annotations

from pathlib import Path

from src.tools.registry import ToolResult

_ALLOWED_DIRS = [
    Path("data").resolve(),
    Path("artifacts").resolve(),
    Path("uploads").resolve(),
    Path(".").resolve(),
]


def _resolve_path(path_str: str) -> Path | None:
    p = Path(path_str).expanduser().resolve()
    for allowed in _ALLOWED_DIRS:
        try:
            p.relative_to(allowed)
            return p
        except ValueError:
            pass
    return None


def file_read(path: str) -> ToolResult:
    resolved = _resolve_path(path)
    if not resolved:
        return ToolResult(False, "", f"Access denied: {path} is outside allowed directories")
    if not resolved.exists():
        return ToolResult(False, "", f"File not found: {path}")
    if not resolved.is_file():
        return ToolResult(False, "", f"Not a file: {path}")
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        if len(content) > 10000:
            content = content[:10000] + "\n\n[...truncated]"
        return ToolResult(True, content)
    except Exception as e:
        return ToolResult(False, "", f"Read failed: {e}")


def file_write(path: str, content: str) -> ToolResult:
    resolved = _resolve_path(path)
    if not resolved:
        return ToolResult(False, "", f"Access denied: {path} is outside allowed directories")
    if len(content) > 50000:
        return ToolResult(False, "", "Content too long (max 50000 chars)")
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Written {len(content)} bytes to {resolved}")
    except Exception as e:
        return ToolResult(False, "", f"Write failed: {e}")


def file_list(directory: str = ".") -> ToolResult:
    resolved = _resolve_path(directory)
    if not resolved:
        return ToolResult(False, "", f"Access denied: {directory} is outside allowed directories")
    if not resolved.is_dir():
        return ToolResult(False, "", f"Not a directory: {directory}")
    try:
        entries = []
        for entry in sorted(resolved.iterdir()):
            suffix = "/" if entry.is_dir() else ""
            size = entry.stat().st_size if entry.is_file() else 0
            entries.append(f"{entry.name}{suffix}  ({size} bytes)" if size else entry.name + suffix)
        return ToolResult(True, "\n".join(entries) if entries else "(empty directory)")
    except Exception as e:
        return ToolResult(False, "", f"List failed: {e}")
