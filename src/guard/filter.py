from __future__ import annotations

import re
import time
from typing import Optional


_SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b\d{16,19}\b", "CREDIT_CARD"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "EMAIL"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "PHONE"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", "CREDIT_CARD"),
    (r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b", "BASE64_KEY"),
]


class PIIFilter:
    def __init__(self, redact: bool = True) -> None:
        self.redact = redact
        self.patterns = _SENSITIVE_PATTERNS

    def scan(self, text: str) -> list[dict]:
        findings: list[dict] = []
        for pattern, label in self.patterns:
            for match in re.finditer(pattern, text):
                findings.append({
                    "type": label,
                    "start": match.start(),
                    "end": match.end(),
                    "value": match.group(),
                })
        return findings

    def sanitize(self, text: str) -> str:
        if not self.redact:
            return text
        result = text
        for pattern, label in self.patterns:
            result = re.sub(pattern, f"[REDACTED_{label}]", result)
        return result


_BLOCKED_INPUT_PATTERNS = [
    (r"(?i)(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|above|prior)\s+instructions", "PROMPT_INJECTION"),
    (r"(?i)(?:you are now|act as|from now on|you're now)\s+(?:an?\s+)?(?:free|unrestricted|unconstrained|unlimited)", "JAILBREAK"),
    (r"(?i)(?:DAN|do anything now|jailbreak|bypass|circumvent)", "JAILBREAK_KEYWORD"),
    (r"<[^>]*system[^>]*>", "TAG_INJECTION"),
    (r"(?i)(?:output|respond|return)\s+(?:raw\s+)?(?:JSON|your\s+prompt|system\s+prompt|instructions)", "PROMPT_LEAK"),
]


class ContentFilter:
    def __init__(self) -> None:
        self.input_patterns = _BLOCKED_INPUT_PATTERNS
        self.blocked_topics: list[str] = []

    def check_input(self, text: str) -> Optional[str]:
        for pattern, label in self.input_patterns:
            if re.search(pattern, text):
                return label
        return None

    def check_output(self, text: str) -> Optional[str]:
        if re.search(r"(?i)(?:api[-\s]?key|secret[-\s]?key|password|token)[:\s]+[\w\-]{16,}", text):
            return "PII_LEAK"
        return None


class RateLimiter:
    def __init__(self, max_requests: int = 30, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[str, list[float]] = {}

    def check(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window
        if key not in self._buckets:
            self._buckets[key] = []
        self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]
        if len(self._buckets[key]) >= self.max_requests:
            return False
        self._buckets[key].append(now)
        return True

    def remaining(self, key: str) -> int:
        now = time.time()
        cutoff = now - self.window
        if key not in self._buckets:
            return self.max_requests
        valid = [t for t in self._buckets[key] if t > cutoff]
        return max(0, self.max_requests - len(valid))

    def reset(self, key: str) -> None:
        self._buckets.pop(key, None)
