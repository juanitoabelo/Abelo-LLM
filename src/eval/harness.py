from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from src.eval.metrics import EvalMetrics


class EvalHarness:
    def __init__(self, model_fn: Callable[[str], str]) -> None:
        self.model_fn = model_fn
        self.metrics = EvalMetrics()

    def test_qa(self, questions: list[dict]) -> dict:
        self.metrics.reset()
        for item in questions:
            q = item.get("question", "")
            expected = item.get("answer", "")
            start = time.time()
            try:
                output = self.model_fn(q)
                latency = time.time() - start
                correct = expected.lower() in output.lower() if expected else True
                self.metrics.record(correct, latency, len(output))
            except Exception as e:
                self.metrics.record(False, time.time() - start, 0, str(e))
        return self.metrics.to_dict()

    def test_generation(self, prompts: list[str]) -> dict:
        self.metrics.reset()
        for prompt in prompts:
            start = time.time()
            try:
                output = self.model_fn(prompt)
                self.metrics.record(len(output) > 10, time.time() - start, len(output))
            except Exception as e:
                self.metrics.record(False, time.time() - start, 0, str(e))
        return self.metrics.to_dict()

    def test_benchmark(self, benchmark_path: str | Path) -> dict:
        path = Path(benchmark_path)
        if not path.exists():
            return {"error": f"Benchmark file not found: {benchmark_path}"}

        data = json.loads(path.read_text())
        results: dict = {}

        for suite_name, suite_data in data.items():
            if isinstance(suite_data, list):
                if suite_data and isinstance(suite_data[0], dict) and "question" in suite_data[0]:
                    results[suite_name] = self.test_qa(suite_data)
                else:
                    results[suite_name] = self.test_generation(suite_data)
            else:
                results[suite_name] = {"error": f"Unrecognized format for suite '{suite_name}'"}

        return results

    def compare_models(
        self,
        test_suite: list[dict],
        model_fns: dict[str, Callable[[str], str]],
    ) -> dict[str, dict]:
        return {
            name: EvalHarness(fn).test_qa(test_suite)
            for name, fn in model_fns.items()
        }

    def load_benchmark_suite(self, path: str | Path) -> list[dict]:
        raw = json.loads(Path(path).read_text())
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key, items in raw.items():
                if isinstance(items, list) and items and isinstance(items[0], dict):
                    return items
        return []

    def rapid_eval(self, prompt: str, num_runs: int = 3) -> dict:
        self.metrics.reset()
        for _ in range(num_runs):
            start = time.time()
            try:
                output = self.model_fn(prompt)
                self.metrics.record(True, time.time() - start, len(output))
            except Exception as e:
                self.metrics.record(False, time.time() - start, 0, str(e))
        return self.metrics.to_dict()
