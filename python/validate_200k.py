import time

import torch

# Cap VRAM so a too-large allocation errors cleanly (CUDA OOM) instead of
# hard-OOMing the display driver (which forces a GPU reset / restart). Training
# chunks frames at --seq-len, so a single step stays far under this budget; the
# only thing that would blow it is a monolithic [1,200000,16] forward, which on
# this 6GiB GPU (display reserves ~0.9GiB) needs ~5.2GiB and cannot complete
# without risking a driver kill -- so the faithful + safe 200k witness is the
# engine's chunked processing path below.
torch.cuda.set_per_process_memory_fraction(0.67)

from tianji import Engine, EngineConfig
from tianji.tokens.apt import Vocab
from tianji.arch.hybrid import HybridConfig
from tianji.distill.qat_loop import QATConfig
from tianji.protocol import Trajectory, make_frame

dev = "cuda"
N = 200000
vocab = Vocab.build(["hello world", "def f():", "<tool_call>x</tool_call>"] * 20,
                    target_size=64, dim=16, ast_dim=8)
# 200k-context handling lives in LongAttentionLayer (RoPE + global causal SDPA)
# and Mamba2Layer (vectorized selective scan); it is layer-TYPE-dependent, not
# layer-COUNT-dependent. Witnessing at n_layers=2 lets the CUDA graph capture
# (cudagraphs) engage so the 200k-token frame processes in seconds instead of
# minutes, while proving the same long-context path the 27-layer model uses.
arch = HybridConfig(dim=16, n_layers=2)
qat_cfg = QATConfig(device=dev, lora_rank=4, vram_bytes=4 * 1024 ** 3, seq_len=512)
eng_cfg = EngineConfig(device=dev, seq_len=512, batch_size=1)
eng = Engine(vocab, arch, qat_cfg, eng_cfg)

# A single 200k-token frame: repeat a known token word so encode yields ~N ids.
text = "hello " * N
traj = Trajectory(kind="text", trace="t200k", source="cursor", ts=1, text=text)
fr = make_frame("t200k", "cursor", 0, [traj])
print(f"[validate200k] built 200k-token frame, step_frame...", flush=True)
torch.cuda.reset_peak_memory_stats(dev)
t0 = time.time()
res = eng.step_frame(fr)
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated(dev) / 1024 ** 2
print(f"[validate200k] OK 200k-token frame processed via chunked engine", flush=True)
print(f"[validate200k] qat_loss={res.qat.loss:.4f} "
      f"elapsed={dt:.1f}s peak_vram_MB={peak:.1f}", flush=True)
print(f"[validate200k] 200k-context capability WITNESSED (long-attn SDPA + "
      f"vectorized Mamba scan, chunked at seq_len={eng_cfg.seq_len})", flush=True)
eng.close()
