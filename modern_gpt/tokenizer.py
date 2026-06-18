"""Tokenizers.

Two options:
  * BPETokenizer  - wraps tiktoken's GPT-2 byte-level BPE (vocab 50257). Use for
                    real corpora.
  * CharTokenizer - a trivial character-level tokenizer, handy for fast,
                    fully-offline experiments (e.g. tiny-shakespeare).
"""
from __future__ import annotations

import json


class BPETokenizer:
    def __init__(self, encoding: str = "gpt2"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": [self.itos[i] for i in range(self.vocab_size)]}, f)

    @classmethod
    def load(cls, path: str) -> CharTokenizer:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])
