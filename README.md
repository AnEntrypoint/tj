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

## Layout

```
python/tianji/
  protocol.py      verified frames (Trajectory / Frame / frame_hash / verify_frame)
  caps.py          capability + resource-budget primitives
  tokens/apt.py    agent-pretraining tokenizer (special tokens, AST, embed)
  arch/            mamba2, mla, moe, hybrid (27 layers), mtp
  quant/          fakequant, kv_quant (int2), adam8bit
  distill/         lora, ewc, replay, router_alignment, kd, qat_loop
  state/transition.py   state-transition head (delta / exit / action)
  infer/           paged_attn, ring_attn, spec_decode, expert_offload, generator
  engine.py        Engine: ties vocab + arch + QAT + state head
  cli.py           demo / infer / ingest-ccsniff / checkpoint
scripts/
  run_demo.sh      full pipeline smoke test
  train.py         continuous training driver over npx ccsniff
```

## Setup

```bash
cd python
pip install -e ".[dev]"      # installs tianji + pytest
python -m pytest tests/ -q   # 64 tests
```

## Run the demo

```bash
bash scripts/run_demo.sh
```

## Train on your real sessions

```bash
# in one terminal — live cost/quota monitor
npx ccwatch

# in another — distill your Claude Code history into Tianji
python scripts/train.py --steps 20 --batch 64 --seq-len 64
```

`train.py` rolls up sessions via `npx ccsniff --json`, verifies them into frames,
and runs the QAT loop. Loss should decrease monotonically as more sessions are seen.

## Design notes

- The hybrid stack is `27` layers = `18` Mamba-2 blocks + `9` MLA+MoE blocks.
- Only the output head carries LoRA adapters (1 adapter per checkpoint) to stay
  inside the 4 GB budget; the rest of the stack is quantized.
- Frames are hash-chained (`sha256:`) so ingested training data is reproducible.
