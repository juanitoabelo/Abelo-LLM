from __future__ import annotations

import re
from typing import Optional

from src.llm.router import LLMRouter


class ContentPlanner:
    def __init__(self) -> None:
        self.llm = LLMRouter()

    def classify_request(self, prompt: str) -> str:
        text = (prompt or "").lower()
        if any(kw in text for kw in ["video", "cinematic", "promo", "trailer", "animation", "motion", "movie"]):
            return "video"
        if any(kw in text for kw in ["infographic", "chart", "timeline", "diagram", "visual summary", "dashboard"]):
            return "infographic"
        if any(kw in text for kw in ["audio", "speech", "voice", "podcast", "narrate", "tts", "sound"]):
            return "audio"
        if any(kw in text for kw in ["image", "poster", "illustration", "visual art", "banner", "cover", "artwork", "draw"]):
            return "image"
        if any(kw in text for kw in ["python", "javascript", "typescript", "html", "css", "function", "api", "app"]):
            return "code"
        if "script" in text and not any(audio_kw in text for audio_kw in ["podcast", "narrate", "voice"]):
            return "code"
        if any(kw in text for kw in ["code", "cli"]):
            return "code"
        return "text"

    async def plan_storyboard(self, prompt: str, scene_count: int = 4, timeout_seconds: int = 30) -> list[dict]:
        import asyncio

        planning_prompt = (
            f"You are a professional storyboard artist. For the concept '{prompt}', "
            f"create {scene_count} distinct scenes. For each scene provide:\n"
            "- title (short, descriptive)\n"
            "- description (visual details)\n"
            "- color palette (two hex colors for gradient)\n"
            "- mood (one word)\n"
            "- accent color (one hex color)\n\n"
            f"Return as a JSON array of objects with keys: title, description, palette (array of 2 hex strings), mood, accent."
        )

        async def _llm_storyboard() -> list[dict] | None:
            full_response = []
            try:
                async for chunk in self.llm.generate(prompt=planning_prompt, temperature=0.7, max_tokens=512, stream=True):
                    full_response.append(chunk)
                text = "".join(full_response)
                import json
                json_match = re.search(r"\[.*\]", text, re.DOTALL)
                if json_match:
                    scenes = json.loads(json_match.group())
                    if isinstance(scenes, list) and len(scenes) > 0:
                        return scenes
            except Exception:
                pass
            return None

        try:
            result = await asyncio.wait_for(_llm_storyboard(), timeout=timeout_seconds)
            if result is not None:
                return result
        except (asyncio.TimeoutError, Exception):
            pass

        return self._fallback_storyboard(prompt, scene_count)

    def _fallback_storyboard(self, prompt: str, scene_count: int) -> list[dict]:
        text = (prompt or "").lower()
        palette = ["#060c1c", "#3d64ff"]
        if "neon" in text:
            palette = ["#080618", "#ff5ca8"]
        elif "sunset" in text:
            palette = ["#2d160e", "#ff763c"]
        elif "futuristic" in text:
            palette = ["#06101e", "#18e2bc"]
        elif "nature" in text:
            palette = ["#0a1f0a", "#4caf50"]
        elif "ocean" in text:
            palette = ["#001a2e", "#0077be"]

        scenes = []
        for i in range(scene_count):
            scenes.append({
                "title": f"{prompt[:40]} — Part {i + 1}",
                "description": f"Scene {i + 1}: Visual exploration of {prompt}",
                "palette": palette,
                "mood": "cinematic" if i % 2 == 0 else "dramatic",
                "accent": f"#{(255 - i * 30):02x}{(180 + i * 15):02x}{(80 + i * 20):02x}",
            })
        return scenes
