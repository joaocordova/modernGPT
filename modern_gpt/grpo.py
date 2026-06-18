"""Group Relative Policy Optimization (Shao et al., DeepSeekMath, 2024).

GRPO is the RL algorithm behind DeepSeek's reasoning models. It drops PPO's
learned value/critic network: for each prompt it samples a *group* of G
completions, scores them with a reward function, and uses the **group-normalized
reward** as the advantage:

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

The policy-gradient objective maximizes  A_i * logp(y_i), with a KL penalty to a
reference model for stability. No critic, low variance, cheap to run.
"""
from __future__ import annotations

import torch

from .rl_utils import per_token_logprob


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Normalize rewards within each group. rewards: (n_groups, group_size)."""
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps)


def grpo_loss(policy, ref, input_ids, target_mask, advantages, beta_kl: float = 0.02):
    """input_ids/target_mask: (B, T) flattened groups; advantages: (B,) per sample.

    Loss = -mean( A * sum_t logp(y_t) )  +  beta_kl * KL(policy || ref)
    """
    tok_logp, mask = per_token_logprob(policy, input_ids, target_mask)
    seq_logp = (tok_logp * mask).sum(-1)
    pg_loss = -(advantages.detach() * seq_logp).mean()

    kl = input_ids.new_zeros(())
    if beta_kl > 0:
        with torch.no_grad():
            ref_logp, _ = per_token_logprob(ref, input_ids, target_mask)
        # token-level k3 KL estimator (unbiased, non-negative), masked + averaged
        diff = ref_logp - tok_logp
        kl_tok = (torch.exp(diff) - diff - 1.0) * mask
        kl = kl_tok.sum() / mask.sum().clamp(min=1)

    loss = pg_loss + beta_kl * kl
    return loss, {"loss": loss.item(), "pg_loss": pg_loss.item(), "kl": float(kl.detach())}


def train_grpo(policy, ref, rollouts, lr=1e-4, beta_kl=0.02, group_size=4):
    """rollouts: list of dicts {input_ids, target_mask, rewards(n_groups,group_size)}."""
    policy.train()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95))
    history = []
    for step, r in enumerate(rollouts):
        adv = group_advantages(r["rewards"]).reshape(-1)
        loss, metrics = grpo_loss(policy, ref, r["input_ids"], r["target_mask"],
                                  adv, beta_kl=beta_kl)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        metrics["step"] = step
        metrics["mean_reward"] = r["rewards"].mean().item()
        history.append(metrics)
    return history
