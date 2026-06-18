"""modernGPT: a decoder-only transformer with a contemporary architecture.

Compared to the original GPT-2 / minGPT this implementation uses the techniques
that define 2023-2025 open models (Llama, Mistral, Qwen, DeepSeek):

  * Rotary Position Embeddings (RoPE) instead of learned absolute positions
  * RMSNorm (pre-norm) instead of LayerNorm
  * SwiGLU feed-forward instead of GELU MLP
  * Grouped-Query Attention (GQA) to shrink the KV cache
  * FlashAttention via torch.nn.functional.scaled_dot_product_attention
  * a streaming KV cache for O(1)-per-token autoregressive decoding
  * an optional Mixture-of-Experts FFN with a load-balancing auxiliary loss

The code is intentionally compact and readable rather than maximally optimized.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


# ----------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability even under autocast, then cast back.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


# ----------------------------------------------------------------------------
# Rotary position embeddings
# ----------------------------------------------------------------------------
def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Return (cos, sin) each of shape (max_seq_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)              # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q, k: (B, n_head, T, head_dim); cos, sin: (T, head_dim)."""
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for GQA. (B, n_kv, T, hd) -> (B, n_kv*n_rep, T, hd)."""
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv, n_rep, T, hd)
        .reshape(B, n_kv * n_rep, T, hd)
    )


# ----------------------------------------------------------------------------
# Attention (GQA + RoPE + SDPA + KV cache)
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.o_proj.RESIDUAL_SCALE = True  # flag for scaled init
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Causal mask only needed during full-sequence (prefill) passes. For
        # single-token incremental decoding the one query attends to all cached keys.
        is_causal = past_kv is None and T > 1
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.resid_drop(self.o_proj(y))
        return y, present


# ----------------------------------------------------------------------------
# Feed-forward: SwiGLU and optional Mixture-of-Experts
# ----------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig, hidden: int | None = None):
        super().__init__()
        hidden = hidden or cfg.resolved_ffn_hidden()
        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.down_proj.RESIDUAL_SCALE = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class MoE(nn.Module):
    """Token-choice top-k MoE with a Switch-Transformer load-balancing loss."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.k = cfg.n_experts_active
        # narrower experts so total params stay comparable to a dense FFN
        expert_hidden = cfg.resolved_ffn_hidden() // 2
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg, hidden=expert_hidden) for _ in range(cfg.n_experts)]
        )

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)                       # (N, C)
        logits = self.gate(x_flat)                      # (N, E)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.k, dim=-1)   # (N, k)
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = topk_idx == e                        # (N, k)
            if not mask.any():
                continue
            token_idx, slot = mask.nonzero(as_tuple=True)
            weight = topk_probs[token_idx, slot].unsqueeze(-1)
            out.index_add_(0, token_idx, self.experts[e](x_flat[token_idx]) * weight)

        # load-balancing aux loss: encourage uniform expert usage.
        # fraction of tokens routed to each expert * mean routing prob to it.
        with torch.no_grad():
            one_hot = F.one_hot(topk_idx, self.n_experts).float().sum(1)  # (N, E)
            frac = one_hot.mean(0)                                        # (E,)
        prob_mean = probs.mean(0)                                         # (E,)
        aux = self.n_experts * (frac * prob_mean).sum()

        return out.reshape(B, T, C), aux


# ----------------------------------------------------------------------------
# Transformer block (pre-norm)
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.use_moe = cfg.use_moe
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = MoE(cfg) if cfg.use_moe else SwiGLU(cfg)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        h, present = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache)
        x = x + h
        mlp_out = self.mlp(self.ffn_norm(x))
        if self.use_moe:
            mlp_out, aux = mlp_out
        else:
            aux = None
        x = x + mlp_out
        return x, present, aux


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------
class ModernGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual output projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False, start_pos=0):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos = self.cos[start_pos:start_pos + T]
        sin = self.sin[start_pos:start_pos + T]

        presents = [] if use_cache else None
        aux_total = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, present, aux = block(x, cos, sin, past, use_cache)
            if use_cache:
                presents.append(present)
            if aux is not None:
                aux_total = aux_total + aux

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            if self.cfg.use_moe:
                loss = loss + self.cfg.moe_aux_coef * aux_total
        return logits, loss, presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """Autoregressive sampling. With use_cache=True, decoding is O(1) per step."""
        self.eval()
        past_kvs = None
        start_pos = 0
        cur = idx
        for _ in range(max_new_tokens):
            # crop to block size on the prefill step if needed
            cur_cropped = cur[:, -self.cfg.block_size:]
            logits, _, presents = self.forward(
                cur_cropped if past_kvs is None else cur[:, -1:],
                use_cache=use_cache,
                past_kvs=past_kvs,
                start_pos=start_pos,
            )
            if use_cache:
                past_kvs = presents
                start_pos = past_kvs[0][0].size(2)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            cur = torch.cat((cur, next_tok), dim=1)
        return cur

    def configure_optimizers(self, weight_decay, lr, betas, device_type, use_muon=True):
        """Build the optimizer(s). With Muon, 2D matrices in blocks use Muon and
        everything else (embeddings, norms, head, biases) uses AdamW."""
        from .optim import build_optimizers
        return build_optimizers(self, weight_decay, lr, betas, device_type, use_muon)
