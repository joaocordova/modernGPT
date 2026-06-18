"""Shared helpers for post-training (SFT / DPO / GRPO).

The key primitive is `sequence_logprob`: the sum of token log-probabilities of a
response under a model, with correct next-token alignment and masking of the
prompt. DPO and GRPO are both thin layers on top of this.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logprob(model, input_ids: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token_t | token_<t) over positions where target_mask == 1.

    input_ids:   (B, T) full sequence (prompt + response)
    target_mask: (B, T) 1 for response tokens to score, 0 for prompt/pad
    returns:     (B,) summed log-probabilities
    """
    logits, _, _ = model(input_ids)
    # logits[:, t] predicts token t+1, so align predictions with input_ids[:, 1:]
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    mask = target_mask[:, 1:].to(logp.dtype)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(-1)


def per_token_logprob(model, input_ids: torch.Tensor, target_mask: torch.Tensor):
    """Like `sequence_logprob` but returns (token_logp, mask) without summing."""
    logits, _, _ = model(input_ids)
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    mask = target_mask[:, 1:].to(logp.dtype)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_logp, mask
