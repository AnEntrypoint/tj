from .lora import (LoRAAdapter, LoRAConfig, wrap_linear_with_lora, wrap_model_with_lora,
                    merge_lora, save_lora_state, load_lora_state)
from .ewc import EWCState, compute_fisher, consolidate, ewc_penalty
from .replay import ReplayBuffer
from .router_alignment import RouterAlignment, apply_router_bias, SOURCE_TO_EXPERT
from .kd import KDConfig, kd_loss, StubTeacher, make_stub_teacher, Teacher
from .qat_loop import QATLoop, QATConfig, QATStepResult
