from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional


class DatasetBuilder:
    FORMATS = {
        "qa": "Q: {question}\nA: {answer}",
        "chat": "<|user|>\n{user}\n<|assistant|>\n{assistant}",
        "instruct": "### Instruction\n{instruction}\n### Response\n{response}",
        "raw": "{text}",
    }

    def __init__(self, output_dir: str | Path = "data/training") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def from_text_files(self, dir_path: str | Path, format: str = "raw") -> list[dict]:
        path = Path(dir_path)
        samples: list[dict] = []
        for f in sorted(path.rglob("*")):
            if f.suffix.lower() in {".txt", ".md"}:
                text = f.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    samples.append({"text": text, "source": str(f)})
        return samples

    def from_jsonl(self, file_path: str | Path) -> list[dict]:
        path = Path(file_path)
        if not path.exists():
            return []
        samples: list[dict] = []
        for line in path.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return samples

    def from_qa_pairs(self, pairs: list[dict], format: str = "qa") -> list[dict]:
        template = self.FORMATS.get(format, self.FORMATS["raw"])
        samples: list[dict] = []
        for pair in pairs:
            if format == "qa":
                samples.append({"text": template.format(question=pair["question"], answer=pair["answer"])})
            elif format == "chat":
                samples.append({"text": template.format(user=pair["user"], assistant=pair["assistant"])})
            elif format == "instruct":
                samples.append({"text": template.format(instruction=pair["instruction"], response=pair["response"])})
        return samples

    def convert_to_jsonl(self, samples: list[dict], output_name: str = "train") -> Path:
        out_path = self.output_dir / f"{output_name}.jsonl"
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps({"text": s["text"]}) + "\n")
        return out_path

    def split(self, samples: list[dict], train_ratio: float = 0.9) -> tuple[list[dict], list[dict]]:
        shuffled = list(samples)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        return shuffled[:split_idx], shuffled[split_idx:]

    def augment_with_ollama(
        self,
        samples: list[dict],
        model: str = "llama3.2:1b",
        field: str = "text",
        max_samples: int = 100,
    ) -> list[dict]:
        import json as _json
        from urllib.request import Request, urlopen

        augmented = list(samples)
        to_augment = [s for s in samples[:max_samples] if s.get(field)]

        for s in to_augment:
            prompt = f"""Generate a variation of the following text that preserves the meaning but rephrases it:

Original: {s[field]}

Variation:"""
            try:
                payload = _json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.8}}).encode()
                req = Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=30) as resp:
                    data = _json.loads(resp.read().decode())
                    variation = data.get("response", "").strip()
                    if variation:
                        augmented.append({field: variation, "source": "ollama_augmented"})
            except Exception:
                continue
        return augmented

    def count_tokens(self, samples: list[dict], text_field: str = "text") -> int:
        total = 0
        for s in samples:
            total += len(s.get(text_field, "")) // 4
        return total
