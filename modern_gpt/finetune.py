"""Supervised fine-tuning (instruction tuning).

SFT trains on (prompt, response) pairs but masks the loss on the prompt tokens,
so the model only learns to *produce* responses, not to memorize prompts. This
is the first post-training step before preference optimization (DPO/GRPO).

Right-padding + causal attention is loss-correct: response tokens never attend
to trailing pad tokens, and pad positions are masked out of the loss.
"""
from __future__ import annotations

import random

import torch

from .rl_utils import sequence_logprob


def build_sft_batch(examples, pad_id: int, device: str = "cpu"):
    """examples: list of (prompt_ids, response_ids). Returns input_ids, target_mask."""
    seqs = [list(p) + list(r) for p, r in examples]
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    target_mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, (p, r) in enumerate(examples):
        full = list(p) + list(r)
        input_ids[i, :len(full)] = torch.tensor(full)
        target_mask[i, len(p):len(full)] = 1   # response tokens only
    return input_ids.to(device), target_mask.to(device)


def sft_loss(model, input_ids, target_mask):
    """Negative mean log-likelihood over response tokens only."""
    seq_logp = sequence_logprob(model, input_ids, target_mask)
    n_resp = target_mask[:, 1:].sum(-1).clamp(min=1)
    return -(seq_logp / n_resp).mean()


def train_sft(model, examples, pad_id, steps=200, lr=1e-3, batch_size=16, device="cpu"):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    history = []
    for step in range(steps):
        batch = random.sample(examples, min(batch_size, len(examples)))
        input_ids, target_mask = build_sft_batch(batch, pad_id, device)
        loss = sft_loss(model, input_ids, target_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append({"step": step, "loss": loss.item()})
    return history


# ---------------------------------------------------------------------------
# A tiny self-contained instruction task: reverse a fixed-length string.
# Vocab is the alphabet plus a few control characters; everything is fixed
# length so no padding is needed.
# ---------------------------------------------------------------------------
def make_reverse_task(n_examples=512, length=5, seed=0):
    rng = random.Random(seed)
    alphabet = "abcdefghij"
    chars = list(alphabet) + ["r", "e", "v", ":", "=", "\n"]  # control tokens reuse letters
    chars = sorted(set(chars))
    stoi = {c: i for i, c in enumerate(chars)}

    def enc(s):
        return [stoi[c] for c in s]

    examples = []
    for _ in range(n_examples):
        s = "".join(rng.choice(alphabet) for _ in range(length))
        prompt = enc(f"rev:{s}=")
        response = enc(s[::-1] + "\n")
        examples.append((prompt, response))
    return examples, len(chars), stoi
