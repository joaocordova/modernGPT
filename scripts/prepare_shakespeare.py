"""Prepare the tiny-shakespeare char-level dataset.

Downloads the corpus if possible; otherwise falls back to a small bundled sample
so the pipeline is fully runnable offline. Writes train.bin / val.bin / meta.json
/ tokenizer.json into data/shakespeare_char/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modern_gpt.tokenizer import CharTokenizer  # noqa: E402

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

FALLBACK = (
    "ROMEO:\nBut soft, what light through yonder window breaks?\n"
    "It is the east, and Juliet is the sun.\n"
    "JULIET:\nO Romeo, Romeo, wherefore art thou Romeo?\n"
    "Deny thy father and refuse thy name;\n"
) * 400


def fetch_text() -> str:
    try:
        import urllib.request
        with urllib.request.urlopen(URL, timeout=10) as r:
            print("[data] downloaded tiny-shakespeare")
            return r.read().decode("utf-8")
    except Exception as e:  # offline -> bundled fallback
        print(f"[data] download failed ({e}); using bundled fallback sample")
        return FALLBACK


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "shakespeare_char")
    os.makedirs(out_dir, exist_ok=True)
    text = fetch_text()
    tok = CharTokenizer.from_text(text)
    tok.save(os.path.join(out_dir, "tokenizer.json"))

    ids = tok.encode(text)
    n = len(ids)
    train_ids = np.array(ids[: int(n * 0.9)], dtype=np.uint16)
    val_ids = np.array(ids[int(n * 0.9):], dtype=np.uint16)
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": tok.vocab_size, "tokenizer": "char"}, f)
    print(f"[data] vocab={tok.vocab_size} train={len(train_ids)} val={len(val_ids)} -> {out_dir}")


if __name__ == "__main__":
    main()
