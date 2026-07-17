from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen


class DistillationTrainer:
    def __init__(
        self,
        teacher_model: str = "qwen3.5:latest",
        output_dir: str | Path = "data/distillation",
    ) -> None:
        self.teacher_model = teacher_model
        self.ollama_url = "http://localhost:11434"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset(
        self,
        seed_texts: list[str],
        num_samples: int = 100,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        samples: list[dict] = []
        batch_size = min(10, max(1, num_samples // len(seed_texts)))

        for seed in seed_texts:
            for i in range(batch_size):
                if len(samples) >= num_samples:
                    break
                sample = self._query_teacher(seed, temperature, system_prompt)
                if sample:
                    samples.append(sample)
                time.sleep(0.1)

        return samples[:num_samples]

    def generate_qa_pairs(
        self,
        topics: list[str],
        num_pairs: int = 50,
    ) -> list[dict]:
        pairs: list[dict] = []
        for topic in topics:
            prompt = f"""Generate a question-answer pair about {topic}.

Format:
Q: <question>
A: <answer>"""
            try:
                payload = json.dumps({
                    "model": self.teacher_model, "prompt": prompt,
                    "stream": False, "options": {"temperature": 0.7},
                }).encode()
                req = Request(f"{self.ollama_url}/api/generate", data=payload, headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                    response = data.get("response", "").strip()
                    lines = response.split("\n")
                    q = ""; a = ""
                    for line in lines:
                        if line.startswith("Q:") or line.startswith("Q:"):
                            q = line[2:].strip()
                        elif line.startswith("A:") or line.startswith("A:"):
                            a = line[2:].strip()
                    if q and a:
                        pairs.append({
                            "instruction": q,
                            "response": a,
                            "topic": topic,
                            "text": f"### Instruction\n{q}\n### Response\n{a}",
                        })
            except Exception:
                continue
        return pairs[:num_pairs]

    def save_jsonl(self, samples: list[dict], filename: str = "distillation_data") -> Path:
        path = self.output_dir / f"{filename}.jsonl"
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        return path

    def run_distillation(
        self,
        data_path: str | Path,
        student_model: str = "llama3.2:1b",
        output_name: str = "distilled_model",
        epochs: int = 3,
    ) -> dict:
        from src.training.lora import LoRATrainer

        trainer = LoRATrainer(backend="auto")
        result = trainer.train(
            base_model=student_model,
            data_path=data_path,
            output_dir=str(self.output_dir / output_name),
            epochs=epochs,
        )
        return result

    def _query_teacher(self, prompt: str, temperature: float, system: Optional[str] = None) -> Optional[dict]:
        try:
            payload: dict = {
                "model": self.teacher_model, "prompt": prompt,
                "stream": False, "options": {"temperature": temperature},
            }
            if system:
                payload["system"] = system
            data = json.dumps(payload).encode()
            req = Request(f"{self.ollama_url}/api/generate", data=data, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                response = result.get("response", "").strip()
                if response:
                    return {
                        "text": response,
                        "prompt": prompt,
                        "teacher": self.teacher_model,
                    }
        except Exception:
            pass
        return None
