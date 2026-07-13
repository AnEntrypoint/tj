# Tianji (天机) — 4GB agentic API-distillation runtime

Tianji distills Claude Code agent sessions into a small hybrid LLM that runs under
a **4 GB VRAM** budget. It combines three ideas:

- **Agentic world-modeling** (à la Qwen AgentWorld): the model learns the latent
  *state* of a coding agent and predicts its next *action* (`tianji.state.transition`).
- **DeepSeek-grade efficiency**: hybrid Mamba-2 + MLA + MoE stack, Multi-Token
  Prediction head, and QAT with int2/int4 quantization + 8-bit Adam
  (`tianji.arch`, `tianji.distill`, `tianji.quant`).
- **Dynamic tokenization from real sessions**: sessions are sniffed live with
  `npx ccsniff`, tokenized with a special-token AST-aware vocabulary
  (`tianji.tokens.apt`), verified into hash-chained frames (`tianji.protocol`),
  and streamed into the QAT loop.

`npx ccwatch` is the cost/quota statusline monitor — run it in a separate terminal
while training to watch spend live.

---

## Setup

```bash
cd python

# Minimal install (training + inference)
pip install -e ".[dev]"

# With Anthropic-compatible API server
pip install -e ".[dev,api]"

# Run tests (64 tests)
python -m pytest tests/ -q
```

---

## CLI reference

All commands are accessible via `python -m tianji.cli <cmd>` or `tianji <cmd>`
after pip install.

| Command | Description |
|---|---|
| `demo` | Build dim-16 27-layer model, run 1 synthetic QAT step, print loss + VRAM |
| `infer` | Generate tokens from a text prompt |
| `ingest-ccsniff` | Convert ccsniff NDJSON frames into verified Frame objects |
| `checkpoint` | Save / load LoRA adapter checkpoint |
| `serve` | Start Anthropic-compatible API server (requires `[api]` extra) |

### `tianji demo`

Quick smoke test — builds the full model stack and runs one training step.

```bash
python -m tianji.cli demo
# [demo] built model with vocab_size=128, layers=27
# [demo] step loss=~0.5 kd=~0.0 vram=~1.2MB bytes
```

### `tianji infer`

Generate tokens from a text prompt.

```bash
python -m tianji.cli infer --prompt "def fib(n): return n" --n 16
```

| Flag | Default | Description |
|---|---|---|
| `--prompt` | `"def fib(n): return n"` | Input text |
| `--n` | `8` | Number of tokens to generate |
| `--dim` | `16` | Model dimension |
| `--layers` | `27` | Number of hybrid layers |

### `tianji ingest-ccsniff`

Parse ccsniff NDJSON into verified Frame objects. Without `--jsonl` it uses a
built-in sample.

```bash
python -m tianji.cli ingest-ccsniff --jsonl sessions.jsonl --batch 32
```

| Flag | Default | Description |
|---|---|---|
| `--jsonl` | *(none)* | Path to ccsniff NDJSON file |
| `--batch` | `32` | Batch size for frame assembly |

### `tianji checkpoint`

Save or load LoRA adapter checkpoint.

```bash
python -m tianji.cli checkpoint save /tmp/tianji.pt
python -m tianji.cli checkpoint load /tmp/tianji.pt
```

### `tianji serve`

Start the Anthropic-compatible API server (defaults to CUDA when available).

```bash
# start on CUDA (auto-detected) with cudagraphs enabled
tianji serve --port 8080
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | HTTP port |
| `--device` | `cpu` | Inference device (`cpu` or `cuda`) |
| `--ckpt` | *(none)* | Path to LoRA checkpoint `.pt` to load |
| `--reload` | *(flag)* | Hot-reload for development |

See [API reference](#api-reference) for request/response formats.

---

## API reference

The Tianji API server implements the **Anthropic Messages API** so any
Anthropic SDK or HTTP client can use it as a drop-in backend. The server also
exposes a legacy completion endpoint and a health check.

### Start the server

```bash
# Install with API extras
pip install -e "python/[api]"

# Start (CPU inference)
tianji serve --port 8080

# With GPU and a trained checkpoint
tianji serve --device cuda --ckpt .tianji_ckpt/qat.pt

# With API key authentication
TIANJI_API_KEY="sk-my-key" tianji serve --port 8080
```

### Endpoints

#### `GET /health`

```http
GET /health
```

```json
{"status":"ok","model_loaded":true,"vram_bytes":1234567,"version":"0.2.0"}
```

#### `GET /v1/models`

```http
GET /v1/models
```

```json
{"data":[{"id":"tianji-4b","object":"model","created":1712345678,"owned_by":"tianji"}]}
```

#### `POST /v1/messages` (Anthropic Messages API)

**Request:**

```json
{
  "model": "tianji-4b",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "system": "You are a helpful assistant.",
  "max_tokens": 64,
  "temperature": 0.7,
  "stream": false
}
```

**Response (non-streaming):**

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "text", "text": "I'm doing well, thanks!"}
  ],
  "model": "tianji-4b",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 8, "output_tokens": 12}
}
```

**Streaming** (`"stream": true`):

Returns [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
with the same event sequence as the Anthropic Messages API:

```
event: message_start
event: content_block_start
event: content_block_delta  (one per token, with "text_delta")
event: content_block_stop
event: message_delta
event: message_stop
```

#### `POST /v1/complete` (legacy)

```json
{
  "model": "tianji-4b",
  "prompt": "def fib(n):",
  "max_tokens": 32,
  "temperature": 1.0,
  "stream": false
}
```

### Client examples

**Python (Anthropic SDK):**

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8080",
    api_key="test",           # matches TIANJI_API_KEY if set
)

response = client.messages.create(
    model="tianji-4b",
    max_tokens=64,
    messages=[{"role": "user", "content": "Write a Python function"}],
)
print(response.content[0].text)
```

**cURL (non-streaming):**

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -d '{
    "model": "tianji-4b",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 16
  }'
```

**cURL (streaming):**

```bash
curl -N http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -d '{
    "model": "tianji-4b",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 16,
    "stream": true
  }'
```

---

## Training

Train on your live Claude Code sessions:

```bash
# Terminal 1 — cost monitor
npx ccwatch

# Terminal 2 — continuous training
python scripts/train.py --steps 20 --batch 64 --seq-len 64
```

### Training arguments

| Flag | Default | Description |
|---|---|---|
| `--steps` | `20` | Number of training steps |
| `--batch` | `64` | Batch size for frame assembly |
| `--dim` | `16` | Model dimension |
| `--ast-dim` | `8` | AST embedding dimension |
| `--layers` | `27` | Number of hybrid layers |
| `--seq-len` | `512` | Token chunk size. Raise to `200000` for full 200k context |
| `--since` | `"1h"` | ccsniff window (e.g. `1h`, `7d`, `24h`) |
| `--limit` | `2000` | Max rows per ccsniff fetch |
| `--save-every` | `5` | Checkpoint interval in steps |
| `--checkpoint-dir` | `.tianji_ckpt` | Checkpoint directory |
| `--resume` | *(flag)* | Resume from latest checkpoint |
| `--device` | *auto* | `"cpu"` or `"cuda"` |

Training is incremental: events are deduplicated by `(sid, ts)` so each event is
trained at most once across runs. Checkpoints include LoRA adapters, state-head
weights, and the deduplication set.

---

## Testing

```bash
cd python

# Run all tests (64 tests)
python -m pytest tests/ -q

# Run a specific test file
python -m pytest tests/test_engine.py -v

# Filter by keyword
python -m pytest tests/ -k "qat"
```

### Test layout

| File | Tests | Coverage |
|---|---|---|
| `tests/test_protocol.py` | 4 | Frame hashing, canonical JSON, verification |
| `tests/test_caps.py` | 5 | Cap minting, resource budget, OOM |
| `tests/test_apt.py` | 8 | Vocab, encode/decode, AST extraction |
| `tests/test_arch.py` | 7 | Mamba2, MLA, MoE, hybrid stack shapes |
| `tests/test_quant.py` | 6 | FakeQuant, int2 pack/unpack, Adam8bit |
| `tests/test_state.py` | 3 | State-transition head forward/simulate |
| `tests/test_distill.py` | 8 | LoRA, replay buffer, KD, QATLoop step |
| `tests/test_infer.py` | 5 | PagedKV, ring attention, speculative decode |
| `tests/test_ingest_ccsniff.py` | 8 | ccsniff NDJSON ingestion pipeline |
| `tests/test_engine.py` | 4 | Engine construction, step_frame, simulate |

---

## Quick start (smoke test)

```bash
cd python
pip install -e ".[dev,api]"

# 1. Unit tests
python -m pytest tests/ -q

# 2. CLI demo (build + 1 step)
python -m tianji.cli demo

# 3. Ingest sample data
python -m tianji.cli ingest-ccsniff --batch 4

# 4. Generate tokens
python -m tianji.cli infer --prompt "def fib(n):" --n 8

# 5. Start API server
tianji serve --port 8080 --device cpu &
sleep 2

# 6. Test API
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"tianji-4b","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'

# 7. Full demo pipeline
bash ../scripts/run_demo.sh

# 8. 200k-context validation (CUDA only)
python validate_200k.py
```

---

## Architecture

```
python/tianji/
  __init__.py      Engine, Vocab public API
  engine.py        Engine — ties vocab + arch + QAT + state head
  cli.py           CLI subcommands (demo, infer, ingest, checkpoint, serve)
  server.py        Anthropic-compatible API server (FastAPI)

  protocol.py      Verified frames (Trajectory / Frame / frame_hash)
  caps.py          Capability + resource-budget primitives

  tokens/apt.py    Agent-pretraining tokenizer (special tokens, AST, embed)

  arch/
    mamba2.py      Mamba-2 SSM layer
    mla.py         Multi-head Latent Attention
    long_attn.py   Long-context RoPE attention (200k tokens)
    moe.py         Mixture-of-Experts (static-shape for CUDA graphs)
    hybrid.py      HybridStack — 27 layers: 18 Mamba-2 + 9 (MLA+MoE)
    mtp.py         Multi-Token Prediction head

  quant/
    fakequant.py   Fake int4 quant (straight-through)
    kv_quant.py    int2 KV cache quant
    adam8bit.py    8-bit Adam optimizer

  distill/
    lora.py        LoRA adapters
    ewc.py         Elastic Weight Consolidation
    replay.py      Replay buffer
    kd.py          Knowledge distillation (stub teacher)
    qat_loop.py    QATLoop — training loop with CUDA graph capture

  state/
    transition.py  State-transition head (delta / exit / action)

  infer/
    paged_attn.py  Paged KV cache
    ring_attn.py   Ring attention
    spec_decode.py Speculative decoding
    expert_offload.py  Expert FIFO offloader
    generator.py   Token generator

  ingest/
    ccsniff.py     ccsniff NDJSON → Frame pipeline

scripts/
  train.py         Continuous training driver over npx ccsniff
  run_demo.sh      Full pipeline smoke test
  run_api.sh       API server launcher
```

---

## Design notes

- **27 layers** = 18 Mamba-2 blocks + 9 MLA+MoE blocks.
- **4 GB VRAM budget** enforced as a hard invariant (`ResourceBudget`).
- Only the output head carries **LoRA adapters** (1 adapter per checkpoint);
  the base stack is quantized.
- **Frames are hash-chained** (`sha256:`) for reproducible training data.
- **State-transition head** is trained: it predicts next event kind and
  exit probability from the pooled latent state.
- **CUDA graphs** capture fwd+bwd+optimizer as a single replayable graph
  (no Triton required — works on Windows).
- **Knowledge distillation** uses a deterministic stub teacher by default
  (KD weight = 0). Supply a real teacher to `QATLoop` to activate.
- **`ccsniff --json`** emits `{ts,iso,sid,parent,cwd,project,role,type,tool,isMeta,text}`.
  `tianji.ingest.ccsniff` maps verbatim to frames.
