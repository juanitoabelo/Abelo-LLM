"""Voice interface - Speech-to-Text and Text-to-Speech.

Uses whisper.cpp for STT and edge-tts or pyttsx3 for TTS.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class SpeechToText:
    def __init__(self, model: str = "base", whisper_path: Optional[str] = None) -> None:
        self.model = model
        self.whisper_path = whisper_path or "whisper"

    def transcribe(self, audio_path: str | Path) -> str:
        try:
            result = subprocess.run(
                [self.whisper_path, str(audio_path), "--model", self.model, "--output-format", "json"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get("text", "").strip()
            return f"[whisper error: {result.stderr[:200]}]"
        except FileNotFoundError:
            return self._fallback_stt(audio_path)
        except Exception as e:
            return f"[STT failed: {e}]"

    def _fallback_stt(self, audio_path: str | Path) -> str:
        import base64
        from src.llm.ollama import OllamaBackend
        import asyncio

        try:
            b64 = base64.b64encode(Path(audio_path).read_bytes()).decode()
            backend = OllamaBackend()
            messages = [{"role": "user", "content": "Transcribe this audio recording precisely. Return only the transcription."}]

            collected = ""
            async def _run():
                nonlocal collected
                async for chunk in backend.chat_raw(messages=messages, model="llama3.2:1b"):
                    if "content" in chunk:
                        collected += chunk["content"]
                    if chunk.get("done"):
                        break
            asyncio.run(_run())
            return collected.strip() or "(empty transcription)"
        except Exception as e:
            return f"[STT fallback failed: {e}]"


class TextToSpeech:
    def __init__(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
            except ImportError:
                self._engine = None
        return self._engine

    def speak(self, text: str) -> Optional[bytes]:
        engine = self._get_engine()
        if engine:
            import tempfile
            import os
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            engine.save_to_file(text[:1000], path)
            engine.runAndWait()
            data = Path(path).read_bytes()
            os.unlink(path)
            return data
        return None

    async def speak_async(self, text: str, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                ["edge-tts", "--text", text[:1000], "--write-media", str(path)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and path.exists():
                return path
        except FileNotFoundError:
            pass

        data = self.speak(text)
        if data:
            path.write_bytes(data)
            return path

        path.write_text(f"[TTS placeholder] {text[:500]}")
        return path
