"""Generate all README figures from the JSON produced by train.py / benchmark.py.

Outputs PNGs into assets/. Re-run after changing any experiment; the README
embeds these files directly.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

BLUE, ORANGE, GREEN, GRAY, RED = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C", "#C44E52"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})


def load(name):
    path = os.path.join(RUNS, name)
    if not os.path.exists(path):
        print(f"[figures] missing {name}; skipping its panel")
        return None
    with open(path) as f:
        return json.load(f)


def report_card():
    muon, adamw = load("muon.json"), load("adamw.json")
    bench = load("benchmark.json")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle("modernGPT - report card", fontsize=16, fontweight="bold", y=0.98)

    # (0,0) pretraining loss
    a = ax[0, 0]
    if muon:
        h = muon["history"]
        it = [p["iter"] for p in h]
        a.plot(it, [p["train"] for p in h], color=BLUE, label="train")
        a.plot(it, [p["val"] for p in h], color=ORANGE, label="val")
        a.legend()
    a.set_title("Pretraining loss (Muon)")
    a.set_xlabel("iteration")
    a.set_ylabel("cross-entropy")

    # (0,1) Muon vs AdamW
    a = ax[0, 1]
    if muon and adamw:
        for run, c, name in [(muon, BLUE, "Muon"), (adamw, GRAY, "AdamW")]:
            h = run["history"]
            a.plot([p["iter"] for p in h], [p["val"] for p in h], color=c, label=name)
        a.legend()
    a.set_title("Optimizer: Muon vs AdamW (val)")
    a.set_xlabel("iteration")
    a.set_ylabel("val loss")

    # (0,2) KV cache throughput
    a = ax[0, 2]
    if bench:
        kv = bench["kv_cache"]
        vals = [kv["uncached"]["tok_per_s"], kv["cached"]["tok_per_s"]]
        bars = a.bar(["no cache", "KV cache"], vals, color=[GRAY, GREEN])
        a.bar_label(bars, fmt="%.0f")
        a.set_title(f"Decoding throughput  ({kv['speedup']:.1f}x speedup)")
        a.set_ylabel("tokens / sec")

    # (1,0) GQA KV-cache memory
    a = ax[1, 0]
    if bench:
        rows = bench["gqa"]["rows"]
        labels = [r["label"] for r in rows]
        mem = [r["kv_cache_MB"] for r in rows]
        colors = [RED if r["label"] == "MHA" else (GREEN if r["label"] == "MQA" else BLUE)
                  for r in rows]
        bars = a.bar(labels, mem, color=colors)
        a.bar_label(bars, fmt="%.0f")
        a.set_title("KV-cache memory vs attention type")
        a.set_ylabel("MB (4k ctx, 32L)")
        a.tick_params(axis="x", rotation=30)

    # (1,1) quantization
    a = ax[1, 1]
    if bench:
        q = bench["quant"]
        bars = a.bar(["fp32", "int8"], [q["fp32_MB"], q["int8_MB"]], color=[GRAY, GREEN])
        a.bar_label(bars, fmt="%.1f MB")
        a.set_title(f"int8 weights: {q['ratio']:.1f}x smaller, cos={q['output_cosine']:.3f}")
        a.set_ylabel("model size (MB)")

    # (1,2) MoE load balance
    a = ax[1, 2]
    if bench:
        m = bench["moe"]
        bars = a.bar(range(m["n_experts"]), m["counts"], color=BLUE)
        ideal = sum(m["counts"]) / m["n_experts"]
        a.axhline(ideal, color=RED, ls="--", label="uniform")
        a.set_title(f"MoE routing (top-{m['top_k']} of {m['n_experts']})")
        a.set_xlabel("expert id")
        a.set_ylabel("tokens routed")
        a.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(ASSETS, "report_card.png"), bbox_inches="tight")
    plt.close(fig)
    print("[figures] report_card.png")


def dpo_figure():
    dpo = load("dpo.json")
    if not dpo:
        return
    h = dpo["history"]
    steps = [p["step"] for p in h]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, [p["reward_margin"] for p in h], color=BLUE)
    ax[0].axhline(0, color=GRAY, ls=":")
    ax[0].set_title("DPO: implicit reward margin (chosen - rejected)")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("margin")
    ax[1].plot(steps, [p["acc"] for p in h], color=GREEN)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("DPO: preference accuracy")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("fraction chosen > rejected")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "dpo_margin.png"), bbox_inches="tight")
    plt.close(fig)
    print("[figures] dpo_margin.png")


def architecture_figure():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=color, ec="#333", lw=1.2, alpha=0.92))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold")

    ax.text(6.5, 3.95, "modernGPT - lifecycle", ha="center", fontsize=14, fontweight="bold")
    stages = [("Pretrain\nRoPE/GQA/SwiGLU\n+ Muon", BLUE), ("SFT\ninstruction tune", GREEN),
              ("DPO / GRPO\npreference opt", ORANGE), ("Quantize\nint8 serving", "#937860"),
              ("Agent\ntool-calling", RED)]
    x = 0.4
    for i, (txt, c) in enumerate(stages):
        box(x, 2.2, 2.1, 1.1, txt, c)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 2.5, 2.75), xytext=(x + 2.1, 2.75),
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.6))
        x += 2.5

    ax.text(0.4, 1.7, "Modern decoder components:", fontsize=10, fontweight="bold")
    comps = ["RoPE positions", "RMSNorm (pre-norm)", "SwiGLU FFN", "Grouped-Query Attn",
             "FlashAttention (SDPA)", "Streaming KV cache", "Mixture-of-Experts", "tied embeddings"]
    cols, bw, bh, gx, gy = 4, 2.85, 0.55, 0.25, 0.2
    x0 = 0.4
    for i, c in enumerate(comps):
        col, row = i % cols, i // cols
        x = x0 + col * (bw + gx)
        y = 0.9 - row * (bh + gy)
        box(x, y, bw, bh, c, "#4C72B0")

    fig.savefig(os.path.join(ASSETS, "architecture.png"), bbox_inches="tight")
    plt.close(fig)
    print("[figures] architecture.png")


if __name__ == "__main__":
    report_card()
    dpo_figure()
    architecture_figure()
    print("[figures] done ->", ASSETS)
