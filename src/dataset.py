import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch.utils.data import Dataset

_TEXT_CLEAN_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    without_control = "".join(ch for ch in normalized if ord(ch) >= 32 or ch in "\n\t")
    collapsed = _TEXT_CLEAN_RE.sub(" ", without_control)
    return collapsed.strip()


class TextDataset(Dataset):
    def __init__(
        self,
        token_ids: List[int],
        block_size: int,
        stride: int = 1,
        max_windows: Optional[int] = None,
    ):
        self.block_size = block_size
        self.examples = []

        for start in range(0, len(token_ids) - block_size, stride):
            x = token_ids[start : start + block_size]
            y = token_ids[start + 1 : start + block_size + 1]
            if len(y) == block_size:
                self.examples.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))
            if max_windows and len(self.examples) >= max_windows:
                break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        tokenizer,
        block_size: int,
        stride: int = 1,
        max_windows: Optional[int] = None,
    ):
        token_ids: List[int] = []
        for text in texts:
            cleaned = clean_text(text)
            if not cleaned:
                continue
            token_ids.extend(tokenizer.encode(cleaned))
        if len(token_ids) < block_size + 1:
            if token_ids:
                repeated = []
                while len(repeated) < block_size + 1:
                    repeated.extend(token_ids)
                token_ids = repeated[: block_size + 1]
            else:
                raise ValueError("Not enough tokenized data to build a dataset. Add more training text or lower block_size.")
        return cls(token_ids, block_size, stride=stride, max_windows=max_windows)

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str,
        tokenizer,
        block_size: int,
        stride: int = 1,
        max_windows: Optional[int] = None,
    ):
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        texts: List[str] = []
        for file_path in sorted(root.glob("**/*.*")):
            if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".text"}:
                texts.append(file_path.read_text(encoding="utf-8", errors="ignore"))
        return cls.from_texts(texts, tokenizer, block_size, stride=stride, max_windows=max_windows)
