"""A tiny, dependency-free inference server (Python stdlib only).

Loads a checkpoint, optionally int8-quantizes it, and serves greedy/sampled
completions over HTTP. Demonstrates the serving stage without pulling in a web
framework.

    python -m modern_gpt.serve --ckpt out/ckpt.pt --data_dir data/shakespeare_char --quantize
    curl -s localhost:8000/generate -d '{"prompt":"ROMEO:","max_new_tokens":80}'
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from .quantize import quantize_int8_
from .sample import build_tokenizer, load_model


def make_handler(model, tok, device):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or "{}")
            ids = torch.tensor([tok.encode(req.get("prompt", "\n"))],
                               dtype=torch.long, device=device)
            out = model.generate(
                ids, int(req.get("max_new_tokens", 64)),
                temperature=float(req.get("temperature", 0.8)),
                top_k=req.get("top_k", 50), use_cache=True,
            )
            body = json.dumps({"completion": tok.decode(out[0].tolist())}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--data_dir", default="data/shakespeare_char")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--quantize", action="store_true", help="int8-quantize before serving")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model, meta = load_model(args.ckpt, args.device)
    if args.quantize:
        quantize_int8_(model)
        print("[serve] int8 quantized")
    tok = build_tokenizer(meta, args.data_dir)
    server = HTTPServer((args.host, args.port), make_handler(model, tok, args.device))
    print(f"[serve] listening on http://{args.host}:{args.port}  (POST /generate)")
    server.serve_forever()


if __name__ == "__main__":
    main()
