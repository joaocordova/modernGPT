"""Run real micro-benchmarks and dump JSON for the README figures.

Outputs runs/benchmark.json and runs/dpo.json. All numbers are measured on the
machine that runs this script (CPU here; the relative trends hold on GPU).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modern_gpt.config import GPTConfig  # noqa: E402
from modern_gpt.dpo import train_dpo  # noqa: E402
from modern_gpt.finetune import make_reverse_task  # noqa: E402
from modern_gpt.model import ModernGPT, MoE  # noqa: E402
from modern_gpt.quantize import model_size_bytes, quantize_int8_  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
os.makedirs(OUT, exist_ok=True)


def bench_kv_cache(n_tokens=96):
    cfg = GPTConfig(vocab_size=128, n_layer=4, n_head=4, n_kv_head=2, n_embd=128,
                    block_size=256)
    model = ModernGPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    out = {}
    for use_cache in (False, True):
        torch.manual_seed(0)
        t0 = time.time()
        with torch.no_grad():
            model.generate(idx, n_tokens, temperature=1.0, use_cache=use_cache)
        dt = time.time() - t0
        out["cached" if use_cache else "uncached"] = {"sec": dt, "tok_per_s": n_tokens / dt}
    out["speedup"] = out["uncached"]["sec"] / out["cached"]["sec"]
    return out


def gqa_kv_memory(n_head=32, head_dim=128, n_layer=32, seq=4096, batch=1, bytes_per=2):
    rows = []
    for n_kv in (n_head, n_head // 2, 8, 4, 2, 1):
        mb = 2 * n_layer * n_kv * head_dim * seq * batch * bytes_per / 1e6
        label = "MHA" if n_kv == n_head else ("MQA" if n_kv == 1 else f"GQA-{n_kv}")
        rows.append({"n_kv_head": n_kv, "label": label, "kv_cache_MB": mb})
    return {"config": {"n_head": n_head, "head_dim": head_dim, "n_layer": n_layer,
                       "seq": seq, "dtype_bytes": bytes_per}, "rows": rows}


def quant_size():
    cfg = GPTConfig(vocab_size=512, n_layer=6, n_head=8, n_kv_head=2, n_embd=384,
                    block_size=256)
    model = ModernGPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        ref, _, _ = model(x)
    fp32 = model_size_bytes(model)
    quantize_int8_(model)
    with torch.no_grad():
        q, _, _ = model(x)
    int8 = model_size_bytes(model)
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), q.flatten(), dim=0).item()
    return {"fp32_MB": fp32 / 1e6, "int8_MB": int8 / 1e6,
            "ratio": fp32 / int8, "output_cosine": cos}


def moe_routing():
    cfg = GPTConfig(vocab_size=256, n_layer=2, n_head=4, n_kv_head=2, n_embd=128,
                    block_size=128, use_moe=True, n_experts=8, n_experts_active=2)
    moe = MoE(cfg).eval()
    torch.manual_seed(0)
    x = torch.randn(4, 64, cfg.n_embd)
    with torch.no_grad():
        logits = moe.gate(x.reshape(-1, cfg.n_embd))
        _, idx = logits.softmax(-1).topk(cfg.n_experts_active, dim=-1)
    counts = torch.bincount(idx.reshape(-1), minlength=cfg.n_experts).tolist()
    return {"n_experts": cfg.n_experts, "top_k": cfg.n_experts_active, "counts": counts}


def dpo_margin_demo():
    """Train DPO to prefer correct string-reversals over corrupted ones; log the margin."""
    torch.manual_seed(0)
    examples, vocab, _ = make_reverse_task(n_examples=64, length=4, seed=1)
    cfg = GPTConfig(vocab_size=vocab, n_layer=3, n_head=4, n_kv_head=2, n_embd=64,
                    block_size=32)
    policy = ModernGPT(cfg)
    ref = copy.deepcopy(policy)
    batches = []
    for p, r in examples:
        good = list(p) + list(r)
        bad = list(p) + list(reversed(r))            # corrupted response
        L = len(good)
        ids_c = torch.tensor([good])
        ids_r = torch.tensor([bad])
        mask = torch.zeros(1, L, dtype=torch.long)
        mask[:, len(p):] = 1
        batches.append({"chosen_ids": ids_c, "chosen_mask": mask,
                        "rejected_ids": ids_r, "rejected_mask": mask.clone()})
    hist = train_dpo(policy, ref, batches * 3, lr=2e-3, beta=0.1)
    return {"history": hist}


def main():
    print("[bench] kv-cache ...")
    kv = bench_kv_cache()
    print("[bench] gqa memory ...")
    gqa = gqa_kv_memory()
    print("[bench] quantization ...")
    q = quant_size()
    print("[bench] moe routing ...")
    moe = moe_routing()
    with open(os.path.join(OUT, "benchmark.json"), "w") as f:
        json.dump({"kv_cache": kv, "gqa": gqa, "quant": q, "moe": moe}, f, indent=2)
    print("[bench] dpo demo ...")
    dpo = dpo_margin_demo()
    with open(os.path.join(OUT, "dpo.json"), "w") as f:
        json.dump(dpo, f, indent=2)
    print(f"[bench] kv speedup={kv['speedup']:.1f}x | quant ratio={q['ratio']:.1f}x "
          f"cos={q['output_cosine']:.3f} | dpo margin {dpo['history'][0]['reward_margin']:.3f}"
          f"->{dpo['history'][-1]['reward_margin']:.3f}")


if __name__ == "__main__":
    main()
