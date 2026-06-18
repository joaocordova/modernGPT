"""Muon optimizer + optimizer construction.

Muon (MomentUm Orthogonalized by Newton-schulz) was introduced by Keller Jordan
et al. (2024) and powers the current nanoGPT speedrun records. It momentum-
averages the gradient of each 2D weight matrix and then *orthogonalizes* the
update via a few Newton-Schulz iterations, which empirically gives a large
sample-efficiency win over AdamW on transformer hidden layers.

Muon is applied only to 2D parameters inside the transformer blocks. Embeddings,
the LM head, RMSNorm gains and any 1D params are trained with AdamW, as
recommended by the authors.
"""
from __future__ import annotations

import torch
from torch.optim import AdamW


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Approximate the orthogonalization (U V^T) of G via a quintic Newton-Schulz
    iteration. Coefficients tuned by Jordan to converge fast in low precision."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    # bf16 is fine and fast on GPU; fall back to fp32 on CPU for portability.
    work_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(work_dtype)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mom, nesterov, ns_steps = group["momentum"], group["nesterov"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                # scale keeps the update RMS roughly invariant to matrix shape
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-group["lr"] * scale)
        return loss


def build_optimizers(model, weight_decay, lr, betas, device_type, use_muon=True):
    """Return a list of optimizers covering all trainable parameters exactly once."""
    muon_params, decay_params, nodecay_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_block = name.startswith("blocks.")
        is_2d = p.ndim == 2
        is_router = "gate.weight" in name  # MoE router: keep on AdamW
        if use_muon and in_block and is_2d and not is_router:
            muon_params.append(p)
        elif p.ndim >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizers = []
    adam_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused = device_type == "cuda"
    optimizers.append(AdamW(adam_groups, lr=lr, betas=betas, fused=fused))
    if muon_params:
        optimizers.append(Muon(muon_params, lr=lr * 2.0, momentum=0.95))

    n_muon = sum(p.numel() for p in muon_params)
    n_adam = sum(p.numel() for p in decay_params + nodecay_params)
    print(f"[optim] Muon params: {n_muon/1e6:.2f}M | AdamW params: {n_adam/1e6:.2f}M")
    return optimizers
