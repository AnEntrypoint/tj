# Tianji (天机) — Agent Instructions

## Testing: Manual Only

No automated pytest suites. All testing is manual via:

```bash
cd python
python manual_debug.py              # all 29 modules, 103 checks
python manual_debug.py --dim 64     # custom dimension
python manual_debug.py --cpu        # force CPU
python manual_debug.py --module qat # single module
python manual_debug.py --list       # list available modules
```

## Exhaustive Debug Checklist

Run `python manual_debug.py` before every change. Expected: 100+ pass, 0 fail.

### Module-by-module verification

| Module | What to verify | Key watchpoints |
|---|---|---|
| `protocol` | Frame hashing, verification, JSON round-trip | Tampered frames rejected, empty frames valid |
| `caps` | Cap minting, regions, budget overflow | OOM raises, closed regions raise |
| `apt` | Vocab build, encode/decode, AST extraction | Padding tokens filtered, OOB handled |
| `mamba2` | Forward shape, state carry, gradient flow | State tensor shape correct, grad flows |
| `mla` | Forward shape, causal masking | Returns tuple (out, kv_compressed) |
| `moe` | Forward shape, aux loss, router bias | Static shape invariant across seq lens |
| `hybrid` | Full stack forward, state carry | 18 Mamba-2 + 9 MLA+MoE = 27 layers |
| `mtp` | Multi-token prediction, speculation | Depth=3, shapes correct |
| `fakequant` | Int4 quant/dequant, error magnitude | Error < 0.2 |
| `kv_quant` | Int2 pack/unpack, error | Error < 1.0 |
| `adam8bit` | Optimizer step, zero grad | Step completes |
| `transition` | Delta, exit, action prediction | 7 event kinds mapped |
| `lora` | Wrap, merge, save/load | Adapter count preserved |
| `ewc` | Fisher computation, penalty, drift | Penalty ~0 for unchanged, >0 after drift |
| `replay` | Push, sample, capacity overflow | Oldest evicted |
| `kd` | Stub teacher, KD loss | Loss > 0 |
| `qat` | Full training step, checkpoint | Loss decreases over steps |
| `contrastive` | InfoNCE, triplet, empty inputs | Empty inputs produce 0 loss |
| `router` | Source bias, apply bias | 5 sources mapped |
| `generator` | Incremental decode, sampling | tok/s benchmarked |
| `paged_attn` | Module importable | Scaffolded |
| `ring_attn` | Module importable | Scaffolded |
| `spec_decode` | Module importable | Scaffolded |
| `expert_offload` | Module importable | Scaffolded |
| `ccsniff` | NDJSON parse, frame assembly, edges | Malformed JSON handled |
| `hf_datasets` | Registry, config resolution | 5 datasets listed |
| `engine` | Frame training, state head, contrastive | Loss decreases, state head predicts |
| `server` | Load/unload, hot-reload | Checkpoint reload works |
| `training` | Script importable | smoke test |

## GPU Setup & VRAM Monitoring

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Monitor VRAM during training
python manual_debug.py --module qat --dim 64  # shows VRAM per step

# Continuous training daemon (low priority)
python scripts/train.py --device cuda --daemon --daemon-poll 30 --dim 64 --precision fp16
```

## Training Debug Commands

```bash
# Full history training (large window)
python scripts/train.py --device cuda --since 999d --dim 64 --precision fp16

# Quick training (recent data only)
python scripts/train.py --device cuda --since 1h --dim 64 --steps 20

# Resume from checkpoint
python scripts/train.py --device cuda --resume --dim 64 --precision fp16

# With HF negative eval
python scripts/train.py --device cuda --hf-dataset the-stack --dim 64
```

## Inference Debug Commands

```bash
# Demo smoke test
python -m tianji.cli demo --device cuda --dim 64

# Text generation
python -m tianji.cli infer --prompt "def fib(n):" --n 16 --dim 64 --device cuda

# With checkpoint
python -c "
from tianji.tokens.apt import Vocab, encode, decode
from tianji.arch.hybrid import HybridConfig
from tianji.distill.qat_loop import QATLoop, QATConfig
from tianji.infer.generator import Generator, GenerateConfig, SamplingConfig
v = Vocab.build(['test']*10, target_size=128, dim=64, ast_dim=8)
q = QATLoop(QATConfig(device='cuda'), HybridConfig(dim=64), vocab_size=128)
q.load_checkpoint('.tianji_ckpt/qat.pt')
g = Generator(q, GenerateConfig(max_tokens=32, sampling=SamplingConfig(temperature=0.7)))
for s in g.generate(encode('def ', v).ids): print(decode([s.token], v), end='', flush=True)
"
```

## API Server Debug Commands

```bash
# Start server
tianji serve --device cuda --dim 64 --vocab-size 128 --ckpt .tianji_ckpt/qat.pt --port 8080

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" -H "x-api-key: test" \
  -d '{"model":"tianji-4b","messages":[{"role":"user","content":"def fib"}],"max_tokens":16}'

# Hot-reload checkpoint
curl -X POST http://localhost:8080/v1/reload \
  -H "Content-Type: application/json" \
  -d '{"ckpt_path":".tianji_ckpt/qat.pt"}'
```

## Common Issues & Fixes

| Issue | Symptom | Fix |
|---|---|---|
| CUDA graph fails | `CUDAGeneratorImpl::current_seed` | Expected on Windows, eager fallback is fine |
| LoRA dim mismatch | `size of tensor a (16) must match size of tensor b (64)` | Pass `--dim 64` to match checkpoint |
| Vocab size mismatch | `size of tensor a (512) must match size of tensor b (128)` | Pass `--vocab-size 128` to match checkpoint |
| ccsniff timeout | Training hangs at step 0 | Reduce `--since` window, avoid `--full-history` for short windows |
| OOM | CUDA out of memory | Reduce `--dim`, use `--precision fp16`, check other GPU processes |
| Server host error | `unrecognized arguments: 0.0.0.0` | Fixed in `cli.py` — `--host` flag properly passed |
| Inference garbage | Output full of `\x01` chars | Fixed in `decode()` — padding tokens filtered |
| Train loss NaN | Loss jumps to NaN | Reduce learning rate, check gradient clipping |

## Architecture Notes

- **27 layers** = 18 Mamba-2 blocks + 9 MLA+MoE blocks
- **4 GB VRAM budget** enforced by `ResourceBudget`
- **CUDA graphs** attempted first, torch.compile secondary, eager fallback
- **AMP fp16** active by default on CUDA
- **Character-level tokenizer** with special tags for agent events
- **State-transition head** predicts delta, exit, and next action from latent state
- **Contrastive loss** separates positive (Claude Code) from negative (public HF) data
- **EWC** prevents catastrophic forgetting across data sources
- **Incremental inference** via Mamba-2 state carry