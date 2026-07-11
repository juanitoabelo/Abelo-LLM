from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.llm.router import LLMRouter


async def generate_audio_artifact(prompt: str, output_path: str | Path, voice: Optional[str] = None) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLMRouter()
    script_parts = []
    system_prompt = "You are a professional narrator. Write a natural, engaging spoken script for the following topic. Keep it conversational and under 200 words."
    async for chunk in llm.generate(
        prompt=f"Write a narration script about: {prompt}",
        system=system_prompt,
        temperature=0.7,
        max_tokens=1024,
        stream=False,
    ):
        script_parts.append(chunk)
    script = "".join(script_parts)

    if path.suffix == ".txt":
        path.write_text(script, encoding="utf-8")
        return path

    try:
        import pyttsx3
        engine = pyttsx3.init()
        if voice:
            voices = engine.getProperty("voices")
            for v in voices:
                if voice.lower() in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
        engine.setProperty("rate", 175)
        engine.save_to_file(script, str(path))
        engine.runAndWait()
    except ImportError:
        script_path = path.with_suffix(".txt")
        script_path.write_text(script, encoding="utf-8")
        print("pyttsx3 not installed. Saved script as text file instead.")
        return script_path

    return path
