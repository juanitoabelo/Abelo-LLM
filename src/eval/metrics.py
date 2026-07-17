from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class EvalMetrics:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
        self.latencies: list[float] = []
        self.output_lengths: list[int] = []
        self.errors: list[str] = []

    def record(self, correct: bool, latency: float, output_length: int, error: Optional[str] = None) -> None:
        self.total += 1
        if correct:
            self.correct += 1
        self.latencies.append(latency)
        self.output_lengths.append(output_length)
        if error:
            self.errors.append(error)

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / max(1, len(self.latencies))

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[len(s) // 2]

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def avg_output_length(self) -> float:
        return sum(self.output_lengths) / max(1, len(self.output_lengths))

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "avg_latency": round(self.avg_latency, 3),
            "p50_latency": round(self.p50_latency, 3),
            "p95_latency": round(self.p95_latency, 3),
            "avg_output_length": round(self.avg_output_length, 1),
            "errors": len(self.errors),
        }
