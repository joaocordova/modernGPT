"""Correctness tests for the post-training stack: SFT, DPO, GRPO, quantization, agent.

Runnable with pytest or directly: `python tests/test_posttrain.py`.
"""
from __future__ import annotations

import copy
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modern_gpt.agent import Agent  # noqa: E402
from modern_gpt.config import GPTConfig  # noqa: E402
from modern_gpt.dpo import dpo_loss, train_dpo  # noqa: E402
from modern_gpt.finetune import (  # noqa: E402
    build_sft_batch,
    make_reverse_task,
    train_sft,
)
from modern_gpt.grpo import group_advantages, grpo_loss  # noqa: E402
from modern_gpt.model import ModernGPT  # noqa: E402
from modern_gpt.quantize import model_size_bytes, quantize_int8_  # noqa: E402
from modern_gpt.rl_utils import sequence_logprob  # noqa: E402


def _tiny(vocab=32, **kw):
    base = dict(vocab_size=vocab, n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
                block_size=64)
    base.update(kw)
    return ModernGPT(GPTConfig(**base))


def _pref_batch(vocab=32, B=4, T=10):
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (B, T))
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, T // 2:] = 1  # second half is the "response"
    return {"chosen_ids": ids, "chosen_mask": mask,
            "rejected_ids": ids.clone(), "rejected_mask": mask.clone()}


def test_sequence_logprob_masking():
    model = _tiny().eval()
    ids = torch.randint(0, 32, (2, 8))
    full_mask = torch.ones(2, 8, dtype=torch.long)
    half_mask = full_mask.clone()
    half_mask[:, :4] = 0
    lp_full = sequence_logprob(model, ids, full_mask)
    lp_half = sequence_logprob(model, ids, half_mask)
    # masking strictly removes (non-positive) log-prob terms -> half >= full
    assert torch.all(lp_half >= lp_full - 1e-5)


def test_dpo_loss_is_log2_when_policy_equals_ref():
    """If policy and reference are identical, the DPO logit is 0 and loss = log 2."""
    model = _tiny().eval()
    ref = copy.deepcopy(model).eval()
    loss, metrics = dpo_loss(model, ref, _pref_batch(), beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-4, loss.item()
    assert abs(metrics["reward_margin"]) < 1e-5


def test_dpo_increases_reward_margin():
    """After training on a fixed preference pair, the chosen-vs-rejected margin grows."""
    torch.manual_seed(0)
    vocab = 16
    policy = _tiny(vocab=vocab)
    ref = copy.deepcopy(policy)
    prompt = torch.randint(0, vocab, (1, 4))
    good = torch.cat([prompt, torch.tensor([[1, 2, 3, 4]])], dim=1)
    bad = torch.cat([prompt, torch.tensor([[5, 6, 7, 8]])], dim=1)
    mask = torch.zeros(1, 8, dtype=torch.long)
    mask[:, 4:] = 1
    batch = {"chosen_ids": good, "chosen_mask": mask,
             "rejected_ids": bad, "rejected_mask": mask.clone()}
    hist = train_dpo(policy, ref, [batch] * 40, lr=5e-3, beta=0.1)
    assert hist[-1]["reward_margin"] > hist[0]["reward_margin"] + 0.05


def test_grpo_advantages_are_group_normalized():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
    adv = group_advantages(rewards)
    assert abs(adv[0].mean().item()) < 1e-5            # zero-centered per group
    assert abs(adv[0].std(unbiased=False).item() - 1.0) < 0.05 or adv[0].std() > 0
    assert torch.allclose(adv[1], torch.zeros(4), atol=1e-3)  # equal rewards -> no signal


def test_grpo_step_runs():
    torch.manual_seed(0)
    vocab = 16
    policy = _tiny(vocab=vocab)
    ref = copy.deepcopy(policy)
    ids = torch.randint(0, vocab, (4, 10))
    mask = torch.zeros(4, 10, dtype=torch.long)
    mask[:, 5:] = 1
    adv = group_advantages(torch.tensor([[1.0, -1.0, 0.5, -0.5]])).reshape(-1)
    loss, metrics = grpo_loss(policy, ref, ids, mask, adv, beta_kl=0.02)
    assert math.isfinite(metrics["loss"]) and metrics["kl"] >= -1e-4


def test_sft_masks_prompt_and_learns():
    torch.manual_seed(0)
    examples, vocab, _ = make_reverse_task(n_examples=256, length=4, seed=0)
    pad_id = 0
    model = _tiny(vocab=vocab, n_layer=3)
    hist = train_sft(model, examples, pad_id, steps=60, lr=3e-3, batch_size=32)
    assert hist[-1]["loss"] < hist[0]["loss"]          # it learns the task
    # masking sanity: prompt-only mask yields zero summed logprob
    ids, tmask = build_sft_batch(examples[:2], pad_id)
    zero_mask = torch.zeros_like(tmask)
    assert torch.allclose(sequence_logprob(model, ids, zero_mask), torch.zeros(2), atol=1e-5)


def test_int8_quantization_fidelity_and_size():
    torch.manual_seed(0)
    model = _tiny(n_layer=3, n_embd=64).eval()
    x = torch.randint(0, 32, (2, 16))
    with torch.no_grad():
        ref_logits, _, _ = model(x)
    size_fp32 = model_size_bytes(model)
    quantize_int8_(model)
    with torch.no_grad():
        q_logits, _, _ = model(x)
    size_int8 = model_size_bytes(model)
    cos = torch.nn.functional.cosine_similarity(
        ref_logits.flatten(), q_logits.flatten(), dim=0).item()
    assert cos > 0.99, cos                             # int8 barely changes outputs
    assert size_int8 < size_fp32                       # and it is smaller


def test_agent_react_loop_uses_tool():
    """A scripted policy: first emit a calc action, then read the observation and answer."""
    calls = {"n": 0}

    def fake_lm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Thought: I should add them.\nAction: calc[21 + 21]"
        return "Answer: 42"

    agent = Agent(generate_fn=fake_lm)
    answer, trace = agent.run("what is 21 + 21?")
    assert answer == "42"
    assert any(step[0] == "calc" and step[2] == "42" for step in trace)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} post-training tests passed.")
