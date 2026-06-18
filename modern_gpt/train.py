"""Training loop for modernGPT.

Features: mixed precision (bf16/fp16 autocast on CUDA), gradient accumulation,
cosine LR schedule with warmup, gradient clipping, periodic eval and
checkpointing. Single-GPU / CPU here; DDP is a small extension (see README).

Example (CPU smoke test):
    python -m modern_gpt.train --preset tiny --data_dir data/shakespeare_char \
        --max_iters 50 --eval_interval 25 --batch_size 8 --device cpu
"""
from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch

from .config import PRESETS, GPTConfig
from .data import get_batch, load_meta
from .model import ModernGPT


def get_lr(it, warmup_iters, lr_decay_iters, max_lr, min_lr):
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, data_dir, eval_iters, batch_size, block_size, device, ctx):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dir, split, batch_size, block_size, device)
            with ctx:
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="tiny", choices=list(PRESETS))
    p.add_argument("--data_dir", default="data/shakespeare_char")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_muon", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--log_json", default="", help="if set, write loss history JSON here")
    args = p.parse_args()
    history = []

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337)

    # autocast context: bf16 if supported, else fp16 on cuda, else no-op
    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
        scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
    else:
        ctx = nullcontext()
        scaler = torch.amp.GradScaler(enabled=False)

    meta = load_meta(args.data_dir)
    cfg = PRESETS[args.preset]
    cfg = GPTConfig(**{**cfg.__dict__, "vocab_size": meta["vocab_size"],
                       "block_size": cfg.block_size})
    model = ModernGPT(cfg).to(device)
    print(f"[model] {cfg.n_layer}L/{cfg.n_head}H (kv={cfg.n_kv_head}) "
          f"d={cfg.n_embd} | params={model.num_params()/1e6:.2f}M | moe={cfg.use_moe}")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizers = model.configure_optimizers(
        args.weight_decay, args.lr, (0.9, 0.95), device_type, use_muon=not args.no_muon
    )

    block_size = cfg.block_size
    best_val = float("inf")
    t0 = time.time()
    X, Y = get_batch(args.data_dir, "train", args.batch_size, block_size, device)

    for it in range(args.max_iters + 1):
        lr = get_lr(it, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = lr

        if it % args.eval_interval == 0:
            losses = estimate_loss(model, args.data_dir, args.eval_iters,
                                   args.batch_size, block_size, device, ctx)
            dt = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {dt:.1f}s")
            history.append({"iter": it, "train": losses["train"], "val": losses["val"],
                            "time": dt})
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                            "meta": meta, "iter": it, "val_loss": best_val},
                           os.path.join(args.out_dir, "ckpt.pt"))

        if it == args.max_iters:
            break

        for _ in range(args.grad_accum):
            with ctx:
                _, loss, _ = model(X, Y)
                loss = loss / args.grad_accum
            X, Y = get_batch(args.data_dir, "train", args.batch_size, block_size, device)
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    print(f"[done] best val loss {best_val:.4f} -> {args.out_dir}/ckpt.pt")
    if args.log_json:
        import json
        os.makedirs(os.path.dirname(args.log_json) or ".", exist_ok=True)
        with open(args.log_json, "w", encoding="utf-8") as f:
            json.dump({"optimizer": "adamw" if args.no_muon else "muon",
                       "preset": args.preset, "history": history}, f, indent=2)
        print(f"[done] loss history -> {args.log_json}")


if __name__ == "__main__":
    main()
