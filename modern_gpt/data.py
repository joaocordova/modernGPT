"""Memory-mapped token dataset, in the style of nanoGPT.

Datasets are stored as flat .bin files of uint16 token ids (train.bin / val.bin)
plus a meta.json describing the tokenizer. get_batch samples random contiguous
windows on the fly, so we never hold the whole corpus in RAM.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def get_batch(data_dir: str, split: str, batch_size: int, block_size: int, device: str):
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def write_bin(ids, path: str) -> None:
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
