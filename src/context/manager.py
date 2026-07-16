from __future__ import annotations

from typing import Optional


# Rough token estimation: ~4 chars per token for most LLMs
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def count_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", "") or "")
        total += 4
    return total


class ContextManager:
    def __init__(self, max_tokens: int = 4096, reserve_tokens: int = 1024) -> None:
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens

    def trim_messages(
        self,
        messages: list[dict],
        extra_text: str = "",
        max_tokens: Optional[int] = None,
    ) -> list[dict]:
        limit = max_tokens or self.max_tokens
        available = limit - estimate_tokens(extra_text) - self.reserve_tokens

        if available <= 0:
            return messages[-2:]

        total = count_messages_tokens(messages)

        if total <= available:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        system_tokens = count_messages_tokens(system_msgs)
        remaining = available - system_tokens

        if remaining <= 0:
            return system_msgs + (non_system[-1:] if non_system else [])

        dropped: list[dict] = []
        while non_system and count_messages_tokens(non_system) > remaining:
            dropped.insert(0, non_system.pop(0))

        result = system_msgs + non_system

        if dropped and estimate_tokens("[...]") <= remaining - count_messages_tokens(non_system):
            summary = self._summarize_dropped(dropped)
            if summary:
                result.insert(len(system_msgs), {"role": "system", "content": f"[Earlier conversation summary: {summary}]"})

        return result

    def _summarize_dropped(self, messages: list[dict]) -> str:
        roles = set(m.get("role", "") for m in messages)
        user_msgs = sum(1 for m in messages if m.get("role") == "user")
        assistant_msgs = sum(1 for m in messages if m.get("role") == "assistant")
        topics = self._extract_topics(messages)
        parts = []
        parts.append(f"{user_msgs} user messages, {assistant_msgs} assistant messages")
        if topics:
            parts.append(f"topics: {', '.join(topics[:5])}")
        return "; ".join(parts)

    def _extract_topics(self, messages: list[dict]) -> list[str]:
        topics: list[str] = []
        for msg in messages:
            content = msg.get("content", "") or ""
            for word in content.split():
                clean = word.strip(".,!?;:'\"()[]{}").lower()
                if len(clean) > 4 and clean[0].isupper():
                    if clean not in topics:
                        topics.append(clean)
        return topics[:8]
