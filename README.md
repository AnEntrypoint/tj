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

`train.py` rolls up sessions via `npx ccsniff --json`, verifies them into
hash-chained frames (`tianji.protocol`), and runs the QAT loop **plus the
state-transition head**. Training is incremental and persistent:

- Each event is trained at most once — rows are deduped by `(sid, ts)` across
  steps and across runs.
- Model + state-head checkpoints are written to `--checkpoint-dir` (default
  `.tianji_ckpt`); resume a prior run with `--resume` to keep accumulating.
- The vocab is seeded from a real sample of the first ingested batch, so
  tokenization covers live agent vocabulary.

The QAT loss (next-token CE + aux) should decrease as more sessions are seen,
and the `state_acc` line reports the state head's next-event-kind prediction
accuracy, which should climb above the 1/7 random baseline as it learns.

## Design notes

- The hybrid stack is `27` layers = `18` Mamba-2 blocks + `9` MLA+MoE blocks.
- Only the output head carries LoRA adapters (1 adapter per checkpoint) to stay
  inside the 4 GB budget; the rest of the stack is quantized. The 4 GB VRAM
  budget is enforced as a hard invariant (`ResourceBudget`) at construction.
- Frames are hash-chained (`sha256:`) so ingested training data is reproducible.
- The state-transition head (`tianji.state.transition`) is **trained**: given the
  latent agent state (pooled hybrid-stack hidden) it predicts the next event
  kind and whether the agent is about to exit. It is wired to its own optimizer.
- Knowledge distillation uses a deterministic stub teacher by default (no real
  teacher is bundled), so KD weight is `0` — only CE + aux train the LM. Supply
  a real teacher to `QATLoop` to activate KD.
- `ccsniff --json` emits `{ts,iso,sid,parent,cwd,project,role,type,tool,isMeta,text}`.
  `tianji.ingest.ccsniff` maps that verbatim; events are grouped by `sid` so a
  frame never mixes sessions. Note: `isError`/`duration` are not part of the
  `--json` contract, so tool-result exit codes are not available from the feed.

