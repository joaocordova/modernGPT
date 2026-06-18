"""Model configuration for modernGPT.

A single dataclass describes the architecture. Defaults target a ~GPT-2 small
footprint but with a modern (Llama/Qwen-era) decoder stack.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- core dims ---
    vocab_size: int = 50304          # GPT-2 BPE padded to a multiple of 64
    n_layer: int = 12
    n_head: int = 12                 # query heads
    n_kv_head: int = 4               # key/value heads (GQA). Must divide n_head.
    n_embd: int = 768
    block_size: int = 1024           # max context length

    # --- feed-forward (SwiGLU) ---
    ffn_hidden: int | None = None    # if None, derived as ~8/3 * n_embd
    multiple_of: int = 64            # round ffn_hidden up to this multiple

    # --- rotary position embeddings ---
    rope_base: float = 10000.0

    # --- regularization / numerics ---
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True      # share token embedding with the output head

    # --- mixture of experts (optional) ---
    use_moe: bool = False
    n_experts: int = 8
    n_experts_active: int = 2        # top-k routing
    moe_aux_coef: float = 0.01       # load-balancing loss weight

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        head_dim = self.n_embd // self.n_head
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def resolved_ffn_hidden(self) -> int:
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        # Llama convention: 2/3 of 4*n_embd, rounded up to `multiple_of`.
        hidden = int(2 * (4 * self.n_embd) / 3)
        return self.multiple_of * ((hidden + self.multiple_of - 1) // self.multiple_of)


# A few ready-made presets.
PRESETS: dict[str, GPTConfig] = {
    "tiny":  GPTConfig(n_layer=4,  n_head=4,  n_kv_head=2, n_embd=128, block_size=256),
    "small": GPTConfig(n_layer=12, n_head=12, n_kv_head=4, n_embd=768, block_size=1024),
    "moe":   GPTConfig(n_layer=6,  n_head=8,  n_kv_head=2, n_embd=512, block_size=512,
                       use_moe=True, n_experts=8, n_experts_active=2),
}
