from __future__ import annotations

from pathlib import Path

from src.llm.router import LLMRouter


async def generate_text_artifact(prompt: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLMRouter()
    content_parts = []
    async for chunk in llm.generate(prompt=prompt, temperature=0.7, max_tokens=2048, stream=False):
        content_parts.append(chunk)
    content = "".join(content_parts)

    path.write_text(content, encoding="utf-8")
    return path
