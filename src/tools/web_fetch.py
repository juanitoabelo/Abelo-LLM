from __future__ import annotations

import re
from urllib.request import Request, urlopen

from src.tools.registry import ToolResult


def web_fetch(url: str) -> ToolResult:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; my_custom_llm/1.0)"
    })
    try:
        with urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type and "html" not in content_type and "json" not in content_type:
                return ToolResult(True, f"[Binary content, type: {content_type}]")
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return ToolResult(False, "", f"Fetch failed: {e}")

    text = _html_to_text(html)
    max_len = 8000
    if len(text) > max_len:
        text = text[:max_len] + "\n\n[...truncated]"
    return ToolResult(True, text.strip())


def _html_to_text(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()
