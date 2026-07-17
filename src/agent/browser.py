"""Web browsing agent — Playwright-based browser automation."""

from __future__ import annotations

import asyncio
import json
from typing import Optional


class BrowserAgent:
    def __init__(self, headless: bool = True, slow_mo: int = 50) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self._playwright = None
        self._browser = None
        self._page = None

    async def _ensure_browser(self):
        if self._page:
            return
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless, slow_mo=self.slow_mo)
        self._page = await self._browser.new_page()
        await self._page.set_viewport_size({"width": 1280, "height": 800})

    async def navigate(self, url: str, wait_until: str = "networkidle") -> dict:
        await self._ensure_browser()
        await self._page.goto(url, wait_until=wait_until)
        return {"url": self._page.url, "title": await self._page.title()}

    async def get_text(self, selector: str = "body") -> str:
        await self._ensure_browser()
        el = await self._page.query_selector(selector)
        if el:
            return await el.inner_text()
        return ""

    async def click(self, selector: str) -> bool:
        await self._ensure_browser()
        try:
            await self._page.click(selector)
            return True
        except Exception:
            return False

    async def fill(self, selector: str, value: str) -> bool:
        await self._ensure_browser()
        try:
            await self._page.fill(selector, value)
            return True
        except Exception:
            return False

    async def extract_links(self) -> list[dict]:
        await self._ensure_browser()
        links = await self._page.evaluate("""
            Array.from(document.querySelectorAll('a')).map(a => ({ text: a.textContent.trim(), href: a.href }))
        """)
        return links[:50]

    async def screenshot(self, path: str = "artifacts/browser_screenshot.png") -> str:
        await self._ensure_browser()
        await self._page.screenshot(path=path, full_page=True)
        return path

    async def execute_js(self, script: str) -> Any:
        await self._ensure_browser()
        return await self._page.evaluate(script)

    async def search_and_extract(self, query: str, search_url: str = "https://duckduckgo.com") -> dict:
        await self._ensure_browser()
        await self._page.goto(search_url, wait_until="networkidle")
        await self._page.fill('input[name="q"]', query)
        await self._page.press('input[name="q"]', "Enter")
        await self._page.wait_for_timeout(2000)
        title = await self._page.title()
        snippets = await self._page.evaluate("""
            Array.from(document.querySelectorAll('.result__body, .result, article, .rnr-snippet'))
                .slice(0, 5).map(el => el.textContent.trim())
        """)
        return {"query": query, "title": title, "results": snippets[:5]}

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None
