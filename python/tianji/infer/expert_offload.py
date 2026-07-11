"""Expert offloader: FIFO eviction keeping a fixed number of experts resident."""
from __future__ import annotations

from collections import OrderedDict
from typing import List


class ExpertOffloader:
    def __init__(self, n_experts: int, n_resident: int):
        self.n_experts = n_experts
        self.n_resident = n_resident
        self.resident: "OrderedDict[int, None]" = OrderedDict()

    def route(self, expert_id: int) -> None:
        if expert_id in self.resident:
            self.resident.move_to_end(expert_id)
            return
        if len(self.resident) >= self.n_resident:
            self.resident.popitem(last=False)
        self.resident[expert_id] = None
