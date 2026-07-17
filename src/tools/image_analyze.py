"""Image analysis tool using vision models."""

from __future__ import annotations

from pathlib import Path

from src.tools.registry import ToolResult


def image_analyze(image_path: str, prompt: str = "Describe this image in detail.") -> ToolResult:
    import asyncio
    from src.vision.analyzer import VisionAnalyzer

    path = Path(image_path)
    if not path.exists():
        return ToolResult(False, "", f"Image not found: {image_path}")

    analyzer = VisionAnalyzer()

    async def _run():
        description = await analyzer.analyze_async(path, prompt)
        return description

    try:
        result = asyncio.run(_run())
        return ToolResult(True, result if result else "(no description generated)")
    except Exception as e:
        return ToolResult(False, "", f"Image analysis failed: {e}")
