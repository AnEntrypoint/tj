"""Anthropic-compatible API server for Tianji-4B.

Implements the Messages API (``/v1/messages``) and model listing
(``/v1/models``) so the runtime can be used as a drop-in backend for any
Anthropic client library, including streaming via SSE.

Quick start:

.. code-block:: bash

    pip install tianji[api]
    python -m tianji.server --port 8080
    curl http://localhost:8080/v1/messages \\
      -H "Content-Type: application/json" \\
      -H "x-api-key: test" \\
      -d '{"model":"tianji-4b","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

try:
    import torch
except ImportError:
    torch = None  # type:ignore[assignment]

from .tokens.apt import Vocab, encode, decode
from .arch.hybrid import HybridConfig
from .distill.qat_loop import QATConfig, QATLoop
from .infer.generator import Generator, GenerateConfig, GenerateStep

# ---------------------------------------------------------------------------
# Schemas – plain dicts to avoid external pydantic dependency
# ---------------------------------------------------------------------------

ROLE_MAP = {"user": "user", "assistant": "assistant", "system": "system"}

# ---------------------------------------------------------------------------
# Model state — loaded once at startup so every request reuses the same QATLoop
# ---------------------------------------------------------------------------

_loop: Optional[QATLoop] = None
_vocab: Optional[Vocab] = None
_generator: Optional[Generator] = None


def _build_vocab() -> Vocab:
    corpus = [
        "def fib(n): return n",
        "<tool_call>{}</tool_call>",
        "<bash_output>ok</bash_output>",
        "<system>agent</system>",
        "<cot>plan</cot>",
        "<diff>--- a\n+++ b\n</diff>",
        "hello world",
        "def main():",
        "import os",
        "print",
        "return",
        "if True:",
    ] * 8
    return Vocab.build(corpus, target_size=512, dim=16, ast_dim=8)


def load_model(device: str = "cpu", ckpt_path: Optional[str] = None,
               reload: bool = False) -> dict:
    global _loop, _vocab, _generator
    if _loop is not None and not reload:
        return {"status": "already loaded", "device": device}

    if reload and _loop is not None:
        _loop.close()
        _loop = None
        _generator = None

    _vocab = _build_vocab()
    arch = HybridConfig(dim=16, n_layers=27)
    qat_cfg = QATConfig(device=device, lora_rank=4, vram_bytes=4 * 1024 ** 3)
    _loop = QATLoop(qat_cfg, arch, vocab_size=_vocab.size)
    _generator = Generator(_loop, GenerateConfig(max_tokens=64, paged_kv_blocks=8))

    if ckpt_path and os.path.exists(ckpt_path):
        n = _loop.load_checkpoint(ckpt_path)
        print(f"[server] loaded LoRA checkpoint ({n} adapters) from {ckpt_path}")

    info = {
        "status": "loaded",
        "device": device,
        "vocab_size": _vocab.size,
        "layers": arch.n_layers,
        "dim": arch.dim,
    }
    print(f"[server] model loaded: {json.dumps(info)}")
    return info


def reload_model(ckpt_path: str) -> dict:
    """Hot-reload checkpoint without restarting the server."""
    global _loop, _generator
    if _loop is None:
        return {"status": "error", "message": "no model loaded"}
    n = _loop.load_checkpoint(ckpt_path)
    print(f"[server] hot-reloaded LoRA checkpoint ({n} adapters) from {ckpt_path}")
    return {"status": "reloaded", "adapters_loaded": n}


def unload_model() -> None:
    global _loop, _vocab, _generator
    if _loop is not None:
        _loop.close()
    _loop = None
    _vocab = None
    _generator = None


def _get_vram() -> int:
    if _loop is not None:
        return int(_loop._vram())
    return 0


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

def _messages_to_text(messages: list[dict], system: Optional[str] = None) -> str:
    parts: list[str] = []
    if system:
        parts.append(f"<system>{system}</system>")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [
                b.get("text", "") for b in content if b.get("type") == "text"
            ]
            content = "\n".join(texts)
        parts.append(f"<{role}>{content}</{role}>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(
    prompt_text: str,
    max_tokens: int = 64,
    temperature: float = 1.0,
) -> tuple[list[int], str]:
    if not _generator or not _vocab:
        raise RuntimeError("model not loaded")

    out = encode(prompt_text, _vocab, parse_ast=True)
    prompt_ids = out.ids[:256]
    n_prompt = len(prompt_ids)

    toks: list[int] = []
    for step in _generator.generate(prompt_ids):
        toks.append(step.token)
        if len(toks) >= max_tokens:
            break

    output_text = decode(toks, _vocab) if toks else ""
    return toks, output_text


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    FastAPI = None  # type:ignore[misc]


def _make_app() -> Optional[FastAPI]:
    if not _HAS_FASTAPI:
        return None

    app = FastAPI(
        title="Tianji-4B",
        description="Anthropic-compatible API for Tianji-4B agentic distillation runtime",
        version="0.2.0",
    )

    # -----------------------------------------------------------------------
    # Middleware: simple API-key check
    # -----------------------------------------------------------------------
    _API_KEY = os.environ.get("TIANJI_API_KEY", "")

    @app.middleware("http")
    async def _check_auth(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        if _API_KEY:
            key = request.headers.get("x-api-key", "")
            if key != _API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"type": "authentication_error", "message": "invalid x-api-key"}},
                )
        return await call_next(request)

    # -----------------------------------------------------------------------
    # GET /health
    # -----------------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": _loop is not None,
            "vram_bytes": _get_vram(),
            "version": "0.2.0",
        }

    # -------------------------------------------------------------------
    # POST /v1/reload — hot-reload checkpoint without restart
    # -------------------------------------------------------------------
    @app.post("/v1/reload")
    async def reload(request: Request):
        if not _loop:
            raise HTTPException(status_code=503, detail="model not loaded")
        body = await request.json()
        ckpt_path = body.get("ckpt_path")
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise HTTPException(status_code=400, detail="ckpt_path required")
        return reload_model(ckpt_path)

    # -----------------------------------------------------------------------
    # GET /v1/models
    # -----------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models():
        return {
            "data": [
                {
                    "id": "tianji-4b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "tianji",
                    "permission": [],
                    "root": "tianji-4b",
                    "parent": None,
                }
            ]
        }

    # -----------------------------------------------------------------------
    # POST /v1/messages  (Anthropic Messages API)
    # -----------------------------------------------------------------------
    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        model = body.get("model", "tianji-4b")
        msgs = body.get("messages", [])
        system = body.get("system", None)
        max_tokens = min(int(body.get("max_tokens", 64)), 1024)
        temperature = float(body.get("temperature", 1.0))
        stream = bool(body.get("stream", False))

        if _loop is None:
            raise HTTPException(503, "model not loaded")

        prompt_text = _messages_to_text(msgs, system)
        n_prompt_ids = len(encode(prompt_text, _vocab, parse_ast=True).ids)

        if stream:
            return StreamingResponse(
                _stream_generate(prompt_text, max_tokens, temperature, model, n_prompt_ids),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        toks, output_text = _generate(prompt_text, max_tokens, temperature)
        return _make_message_response(model, n_prompt_ids, len(toks), output_text)

    # -----------------------------------------------------------------------
    # POST /v1/complete  (legacy completion endpoint)
    # -----------------------------------------------------------------------
    @app.post("/v1/complete")
    async def complete(request: Request):
        body = await request.json()
        model = body.get("model", "tianji-4b")
        prompt = body.get("prompt", "")
        max_tokens = min(int(body.get("max_tokens", 64)), 1024)
        temperature = float(body.get("temperature", 1.0))
        stream = bool(body.get("stream", False))

        if _loop is None:
            raise HTTPException(503, "model not loaded")

        n_prompt_ids = len(encode(prompt, _vocab, parse_ast=True).ids)

        if stream:
            return StreamingResponse(
                _stream_generate(prompt, max_tokens, temperature, model, n_prompt_ids),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        toks, output_text = _generate(prompt, max_tokens, temperature)
        return {
            "id": f"cmpl_{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "text": output_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": n_prompt_ids,
                "completion_tokens": len(toks),
                "total_tokens": n_prompt_ids + len(toks),
            },
        }

    return app


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

async def _stream_generate(
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    model: str,
    n_prompt_ids: int,
) -> AsyncGenerator[str, None]:
    if not _generator or not _vocab:
        yield f"data: {json.dumps({'error':'model not loaded'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    out = encode(prompt_text, _vocab, parse_ast=True)
    prompt_ids = out.ids[:256]

    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    content_block_id = f"cb_{uuid.uuid4().hex[:8]}"

    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':model,'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':n_prompt_ids,'output_tokens':0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"

    collected: list[str] = []
    for step in _generator.generate(prompt_ids):
        step_token = step.token
        token_text = decode([step_token], _vocab)
        collected.append(token_text)
        yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':token_text}})}\n\n"
        if len(collected) >= max_tokens:
            break
        await asyncio.sleep(0)

    n_out = len(collected)
    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':n_out}})}\n\n"
    yield "event: message_stop\ndata: {}\n\n"


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def _make_message_response(
    model: str,
    n_prompt: int,
    n_out: int,
    output_text: str,
) -> dict:
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": output_text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": n_prompt,
            "output_tokens": n_out,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def serve(argv: Optional[list[str]] = None) -> int:
    """``tianji serve`` subcommand."""
    import argparse

    _default_dev = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    ap = argparse.ArgumentParser(prog="tianji serve")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default=_default_dev, choices=["cpu", "cuda"])
    ap.add_argument("--ckpt", default=None, help="path to LoRA checkpoint .pt")
    ap.add_argument("--reload", action="store_true", help="hot-reload (dev only)")
    args = ap.parse_args(argv)

    load_model(device=args.device, ckpt_path=args.ckpt)

    app = _make_app()
    if app is None:
        print("install fastapi + uvicorn: pip install tianji[api]", file=sys.stderr)
        return 1

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


# ---------------------------------------------------------------------------
# ``python -m tianji.server``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    raise SystemExit(serve(sys.argv[1:]))
