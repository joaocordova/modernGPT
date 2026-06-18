"""Direct Preference Optimization (Rafailov et al., 2023), from scratch.

DPO replaces the RLHF pipeline (reward model + PPO) with a single closed-form
classification loss on preference pairs. The policy is nudged to raise the
log-probability of the *chosen* response relative to the *rejected* one, while a
frozen reference model keeps it from drifting too far:

    L = -log sigma( beta * [ (logp_pi(y_w) - logp_ref(y_w))
                            - (logp_pi(y_l) - logp_ref(y_l)) ] )

Sanity check baked into the tests: when policy == reference, every term cancels,
the logit is 0, and the loss equals -log sigma(0) = log 2 ~= 0.6931.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .rl_utils import sequence_logprob


def dpo_loss(policy, ref, batch, beta: float = 0.1):
    """batch: dict with chosen_ids, chosen_mask, rejected_ids, rejected_mask."""
    pi_chosen = sequence_logprob(policy, batch["chosen_ids"], batch["chosen_mask"])
    pi_rejected = sequence_logprob(policy, batch["rejected_ids"], batch["rejected_mask"])
    with torch.no_grad():
        ref_chosen = sequence_logprob(ref, batch["chosen_ids"], batch["chosen_mask"])
        ref_rejected = sequence_logprob(ref, batch["rejected_ids"], batch["rejected_mask"])

    pi_logratio = pi_chosen - pi_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = beta * (pi_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()

    # implicit rewards (for logging / the margin plot)
    chosen_reward = beta * (pi_chosen - ref_chosen).detach()
    rejected_reward = beta * (pi_rejected - ref_rejected).detach()
    metrics = {
        "loss": loss.item(),
        "reward_margin": (chosen_reward - rejected_reward).mean().item(),
        "acc": (chosen_reward > rejected_reward).float().mean().item(),
    }
    return loss, metrics


def train_dpo(policy, ref, batches, lr=1e-4, beta=0.1, device="cpu"):
    """Minimal DPO loop over a list of pre-built batches. Returns metric history."""
    policy.train()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95))
    history = []
    for step, batch in enumerate(batches):
        loss, metrics = dpo_loss(policy, ref, batch, beta=beta)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        metrics["step"] = step
        history.append(metrics)
    return history
