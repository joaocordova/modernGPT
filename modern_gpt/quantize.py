"""Weight-only int8 post-training quantization (per-output-channel, symmetric).

Each Linear weight row is scaled by max(|w|)/127 and stored as int8 plus an fp32
scale, cutting weight memory ~4x. The forward pass dequantizes on the fly. This
is the simplest member of the family that includes GPTQ/AWQ (int4) and is the
standard first lever for cheaper LLM serving.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    def __init__(self, qweight, scale, bias=None):
        super().__init__()
        self.register_buffer("qweight", qweight)   # (out, in) int8
        self.register_buffer("scale", scale)       # (out, 1) fp32
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, lin: nn.Linear):
        W = lin.weight.data.float()
        scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        qweight = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
        bias = lin.bias.data.clone() if lin.bias is not None else None
        return cls(qweight, scale, bias)

    def forward(self, x):
        W = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, W, self.bias)


def quantize_int8_(model: nn.Module, skip_substr=("lm_head",)) -> nn.Module:
    """In-place replace eligible nn.Linear modules with int8 QuantizedLinear.

    The output head is skipped by default (it is the most quality-sensitive and,
    when tied to embeddings, is best left in full precision)."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip_substr):
                setattr(module, child_name, QuantizedLinear.from_linear(child))
    return model


def model_size_bytes(model: nn.Module) -> int:
    """Approximate parameter+buffer storage in bytes."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total
