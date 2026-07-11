from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple
import tempfile
import os


class BPETokenizer:
    """A tiny byte-pair encoding tokenizer for small-scale experiments."""

    def __init__(self, target_vocab_size: int = 256) -> None:
        self.target_vocab_size = target_vocab_size
        self.vocab_size = 0
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._merges: list[tuple[str, str]] = []
        self._initialized = False
        self.unk_token = "<unk>"
        self.unk_token_id = -1

    @staticmethod
    def _get_pair_counts(vocab: Counter[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += freq
        return pair_counts

    @staticmethod
    def _merge_vocab(vocab: Counter[tuple[str, ...], int], pair: tuple[str, str]) -> Counter[tuple[str, ...], int]:
        merged_vocab: Counter[tuple[str, ...], int] = Counter()
        pair_first, pair_second = pair

        for word, freq in vocab.items():
            merged_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair_first and word[i + 1] == pair_second:
                    merged_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            merged_vocab[tuple(merged_word)] += freq

        return merged_vocab

    def fit(self, texts: List[str]) -> None:
        if not texts:
            raise ValueError("At least one text is required to fit the tokenizer")

        corpus: Counter[tuple[str, ...], int] = Counter()
        for text in texts:
            if not text:
                continue
            corpus[tuple(text) + ("</w>",)] += 1

        self._merges = []
        unique_tokens = {token for word in corpus for token in word}
        max_merges = max(0, self.target_vocab_size - len(unique_tokens) - 1)

        for _ in range(max_merges):
            pair_counts = self._get_pair_counts(corpus)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            self._merges.append(best_pair)
            corpus = self._merge_vocab(corpus, best_pair)

        token_set = {token for word in corpus for token in word}
        token_set.discard("</w>")
        sorted_tokens = sorted(token_set)
        sorted_tokens.append("</w>")
        sorted_tokens.append(self.unk_token)

        self._token_to_id = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.unk_token_id = self._token_to_id[self.unk_token]
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}
        self.vocab_size = len(self._token_to_id)
        self._initialized = True

    def _encode_text(self, text: str) -> list[str]:
        tokens = list(text) + ["</w>"]
        while True:
            pair_positions = { (tokens[i], tokens[i + 1]): i for i in range(len(tokens) - 1) }
            merge_pair = next((pair for pair in self._merges if pair in pair_positions), None)
            if merge_pair is None:
                break
            position = pair_positions[merge_pair]
            tokens = tokens[:position] + [tokens[position] + tokens[position + 1]] + tokens[position + 2:]
        return tokens

    def encode(self, text: str) -> List[int]:
        if not self._initialized:
            raise RuntimeError("Tokenizer must be fitted before encoding")

        token_ids: list[int] = []
        for token in self._encode_text(text):
            token_ids.append(self._token_to_id.get(token, self.unk_token_id))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        if not self._initialized:
            raise RuntimeError("Tokenizer must be fitted before decoding")

        decoded = "".join(self._id_to_token.get(token_id, self.unk_token) for token_id in token_ids)
        return decoded.replace("</w>", "")

    def save(self, path: str | Path) -> None:
        data = {
            "target_vocab_size": self.target_vocab_size,
            "token_to_id": self._token_to_id,
            "merges": [list(pair) for pair in self._merges],
            "unk_token": self.unk_token,
            "unk_token_id": self.unk_token_id,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.target_vocab_size = payload["target_vocab_size"]
        self._token_to_id = {token: idx for token, idx in payload["token_to_id"].items()}
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}
        self._merges = [tuple(pair) for pair in payload["merges"]]
        self.unk_token = payload["unk_token"]
        self.unk_token_id = payload["unk_token_id"]
        self.vocab_size = len(self._token_to_id)
        self._initialized = True


# Optional SentencePiece tokenizer wrapper. If sentencepiece is not installed
# the class will raise an informative error when used.
try:
    import sentencepiece as spm  # type: ignore


    class SentencePieceTokenizer:
        """Wrapper around SentencePiece for tokenization + model training."""

        def __init__(self, model_prefix: str = "spm", vocab_size: int = 8000) -> None:
            self.model_prefix = model_prefix
            self.vocab_size = vocab_size
            self.model_file: str | None = None

        def fit(self, texts: List[str]) -> None:
            if not texts:
                raise ValueError("At least one text is required to fit the tokenizer")
            # write temporary corpus file required by SentencePiece
            fd, temp_path = tempfile.mkstemp(prefix="spm_corpus_", suffix=".txt")
            os.close(fd)
            with open(temp_path, "w", encoding="utf-8") as fh:
                for t in texts:
                    fh.write(t.replace("\n", " ") + "\n")

            model_prefix = f"{self.model_prefix}"
            spm.SentencePieceTrainer.Train(
                input=temp_path,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size,
                user_defined_symbols=["<unk>"],
            )
            self.model_file = f"{model_prefix}.model"
            os.remove(temp_path)

        def save(self, path: str | Path) -> None:
            if self.model_file is None or not Path(self.model_file).exists():
                raise RuntimeError("SentencePiece model not trained yet")
            Path(path).write_bytes(Path(self.model_file).read_bytes())

        def load(self, path: str | Path) -> None:
            dest = Path(self.model_prefix + ".model")
            dest.write_bytes(Path(path).read_bytes())
            self.model_file = str(dest)

        def encode(self, text: str) -> List[int]:
            if not self.model_file:
                raise RuntimeError("SentencePiece model not loaded")
            sp = spm.SentencePieceProcessor()
            sp.Load(self.model_file)
            return sp.EncodeAsIds(text)

        def decode(self, token_ids: List[int]) -> str:
            if not self.model_file:
                raise RuntimeError("SentencePiece model not loaded")
            sp = spm.SentencePieceProcessor()
            sp.Load(self.model_file)
            return sp.DecodeIds(token_ids)

except Exception:
    SentencePieceTokenizer = None
