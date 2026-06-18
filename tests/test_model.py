"""Correctness tests. Runnable with pytest, or directly: `python tests/test_model.py`.

The headline test (test_kv_cache_matches_full_forward) proves the streaming KV
cache produces *identical* logits to a full forward pass -- the single most
common place a from-scratch transformer is subtly wrong.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modern_gpt.config import GPTConfig  # noqa: E402
from modern_gpt.model import ModernGPT  # noqa: E402
from modern_gpt.optim import Muon, zeropower_via_newtonschulz5  # noqa: E402


def _tiny(**kw):
    base = dict(vocab_size=64, n_layer=3, n_head=4, n_kv_head=2, n_embd=32,
                block_size=64)
    base.update(kw)
    return GPTConfig(**base)


def test_forward_shapes_and_loss():
    cfg = _tiny()
    model = ModernGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, _ = model(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = _tiny()
    model = ModernGPT(cfg).eval()
    T = 20
    idx = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        full_logits, _, _ = model(idx)            # (1, T, V)

        # replay the same sequence token-by-token through the cache
        past = None
        start = 0
        cached = []
        for t in range(T):
            step_logits, _, past = model(idx[:, t:t + 1], use_cache=True,
                                         past_kvs=past, start_pos=start)
            start += 1
            cached.append(step_logits[:, -1, :])
        cached = torch.stack(cached, dim=1)        # (1, T, V)

    assert torch.allclose(full_logits, cached, atol=1e-4), \
        (full_logits - cached).abs().max().item()


def test_causality():
    """Logits at position t must not depend on tokens after t."""
    torch.manual_seed(0)
    cfg = _tiny()
    model = ModernGPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        a, _, _ = model(idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % cfg.vocab_size  # perturb last token
        b, _, _ = model(idx2)
    # all positions except the last must be unchanged
    assert torch.allclose(a[:, :-1, :], b[:, :-1, :], atol=1e-5)


def test_gqa_param_reduction():
    """GQA must produce smaller K/V projections than full multi-head attention."""
    gqa = ModernGPT(_tiny(n_head=4, n_kv_head=1))
    mha = ModernGPT(_tiny(n_head=4, n_kv_head=4))
    assert gqa.num_params() < mha.num_params()


def test_moe_forward_and_aux():
    cfg = _tiny(use_moe=True, n_experts=4, n_experts_active=2)
    model = ModernGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, _ = model(idx, targets=idx)
    assert torch.isfinite(loss)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_muon_orthogonalization():
    """Newton-Schulz output should be approximately orthogonal (near-isometry)."""
    torch.manual_seed(0)
    G = torch.randn(32, 16)
    raw_spread = torch.linalg.svdvals(G)
    O = zeropower_via_newtonschulz5(G, steps=5)
    # raw Gaussian singular values are widely spread; after orthogonalization
    # they cluster tightly near 1 (the quintic NS5 is an approximation, so we
    # allow a modest band rather than demanding exactly 1).
    s = torch.linalg.svdvals(O)
    assert raw_spread.max() / raw_spread.min() > 3.0          # raw is ill-conditioned
    assert (s.max() < 1.5) and (s.min() > 0.5), (s.min().item(), s.max().item())
    assert s.max() / s.min() < 2.0                            # orthogonalized is well-conditioned


def test_muon_step_runs():
    cfg = _tiny()
    model = ModernGPT(cfg)
    opts = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu", use_muon=True)
    assert any(isinstance(o, Muon) for o in opts)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss, _ = model(idx, targets=idx)
    loss.backward()
    for o in opts:
        o.step()
        o.zero_grad()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
