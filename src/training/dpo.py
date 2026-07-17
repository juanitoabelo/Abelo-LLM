"""DPO (Direct Preference Optimization) training pipeline.

Collects preference pairs from user feedback and prepares DPO training data
for fine-tuning via Ollama Modelfiles or external DPO frameworks.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional


class DPOTrainer:
    def __init__(self, feedback_db: str | Path = "data/feedback.db", output_dir: str | Path = "data/dpo") -> None:
        self.feedback_db = Path(feedback_db)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_preference_pairs(self, limit: int = 1000) -> list[dict]:
        conn = sqlite3.connect(str(self.feedback_db))
        conn.row_factory = sqlite3.Row
        try:
            feedback_rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            pref_rows = conn.execute(
                "SELECT * FROM preferences ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()

        pairs = []
        by_prompt: dict[str, dict[str, str]] = {}
        for fb in feedback_rows:
            prompt = fb["user_message"]
            if prompt not in by_prompt:
                by_prompt[prompt] = {}
            response = fb["assistant_response"]
            if fb["rating"] == 1:
                by_prompt[prompt]["chosen"] = response
                by_prompt[prompt].setdefault("model", fb["model"])
                by_prompt[prompt].setdefault("timestamp", fb["created_at"])
            elif fb["rating"] == -1:
                by_prompt[prompt]["rejected"] = response
                by_prompt[prompt].setdefault("model", fb["model"])
                by_prompt[prompt].setdefault("timestamp", fb["created_at"])
        for prompt, data in by_prompt.items():
            if data.get("chosen") and data.get("rejected"):
                pairs.append({
                    "prompt": prompt,
                    "chosen": data["chosen"],
                    "rejected": data["rejected"],
                    "source": "feedback",
                    "model": data.get("model", ""),
                    "timestamp": data.get("timestamp", 0.0),
                })

        for pref in pref_rows:
            pairs.append({
                "prompt": pref["prompt"],
                "chosen": pref["chosen"],
                "rejected": pref["rejected"],
                "source": "explicit_preference",
                "model": pref["model"],
                "timestamp": pref["created_at"],
            })

        valid = [p for p in pairs if p["chosen"] and p["rejected"]]
        output_path = self.output_dir / f"dpo_pairs_{int(time.time())}.json"
        output_path.write_text(json.dumps(valid, indent=2))
        return valid

    def format_for_trl(self, pairs: list[dict]) -> str:
        """Format as JSONL for Hugging Face TRL DPO trainer."""
        lines = []
        for p in pairs:
            lines.append(json.dumps({
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }))
        output_path = self.output_dir / f"dpo_trl_{int(time.time())}.jsonl"
        output_path.write_text("\n".join(lines))
        return str(output_path)

    def format_for_modelfile(self, pairs: list[dict]) -> str:
        """Format as Modelfile messages for Ollama-based DPO."""
        lines = ["FROM llama3.2:1b", ""]
        for p in pairs[:5]:
            lines.append(f'MESSAGE user "{p["prompt"]}"')
            lines.append(f'MESSAGE assistant "{p["chosen"]}"')
            lines.append("")
        output_path = self.output_dir / f"dpo_modelfile_{int(time.time())}.txt"
        output_path.write_text("\n".join(lines))
        return str(output_path)

    def get_stats(self) -> dict:
        conn = sqlite3.connect(str(self.feedback_db))
        try:
            feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            pref_count = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
            up_count = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = 1").fetchone()[0]
            return {
                "total_feedback": feedback_count,
                "total_preferences": pref_count,
                "thumbs_up": up_count,
                "usable_pairs": min(up_count, feedback_count - up_count),
            }
        finally:
            conn.close()
