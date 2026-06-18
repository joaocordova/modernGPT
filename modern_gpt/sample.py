"""Sample from a trained checkpoint, using the KV cache for fast decoding."""
from __future__ import annotations

import argparse
import os

import torch

from .config import GPTConfig
from .model import ModernGPT
from .tokenizer import BPETokenizer, CharTokenizer


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = ModernGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["meta"]


def build_tokenizer(meta, data_dir):
    if meta["tokenizer"] == "char":
        return CharTokenizer.load(os.path.join(data_dir, "tokenizer.json"))
    return BPETokenizer(meta.get("encoding", "gpt2"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--data_dir", default="data/shakespeare_char")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, meta = load_model(args.ckpt, device)
    tok = build_tokenizer(meta, args.data_dir)

    ids = tok.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature,
                         top_k=args.top_k, use_cache=True)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
