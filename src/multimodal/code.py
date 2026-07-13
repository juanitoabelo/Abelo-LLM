from __future__ import annotations

import re
from pathlib import Path

from src.llm.router import LLMRouter


async def generate_code_artifact(prompt: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    language = "python"
    if re.search(r"\b(js|javascript|node|web|html|css)\b", prompt, re.I):
        language = "javascript"
    elif re.search(r"\b(ts|typescript)\b", prompt, re.I):
        language = "typescript"
    elif re.search(r"\b(rust|rs)\b", prompt, re.I):
        language = "rust"
    elif re.search(r"\b(go|golang)\b", prompt, re.I):
        language = "go"

    system_prompt = (
        f"You are an expert {language} developer. Generate production-quality {language} code "
        f"for the following request. Include proper error handling, typing, and documentation. "
        f"Return ONLY the code block, no explanation."
    )

    llm = LLMRouter()
    code_parts = []
    async for chunk in llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write {language} code for: {prompt}"},
        ],
        temperature=0.3,
        max_tokens=256,
        stream=True,
    ):
        code_parts.append(chunk)
    code = "".join(code_parts)

    code = re.sub(r"^```[\w]*\n", "", code.strip())
    code = re.sub(r"\n```$", "", code.strip())

    path.write_text(code, encoding="utf-8")
    return path
