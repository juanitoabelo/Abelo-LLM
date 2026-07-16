from __future__ import annotations

import json
from urllib.request import Request, urlopen
from urllib.parse import quote

from src.tools.registry import ToolResult


def web_search(query: str) -> ToolResult:
    duckduckgo_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    req = Request(duckduckgo_url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; my_custom_llm/1.0)"
    })
    try:
        with urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return ToolResult(False, "", f"Search failed: {e}")

    results = _parse_ddg_results(html)
    if not results:
        return ToolResult(True, "No search results found.")
    output = ""
    for i, r in enumerate(results[:8], 1):
        output += f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n\n"
    return ToolResult(True, output.strip())


def _parse_ddg_results(html: str) -> list[dict]:
    results: list[dict] = []
    import re
    snippets = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )
    bodies = re.findall(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )
    for i, (url_raw, title_raw) in enumerate(snippets):
        title = re.sub(r"<[^>]+>", "", title_raw).strip()
        url = url_raw.replace("//duckduckgo.com/l/?uddg=", "")
        url = url.split("&")[0] if "&" in url else url
        snippet = ""
        if i < len(bodies):
            snippet = re.sub(r"<[^>]+>", "", bodies[i]).strip()
        results.append({"title": title, "url": url, "snippet": snippet})
    return results
