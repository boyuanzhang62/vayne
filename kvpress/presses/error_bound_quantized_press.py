from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

@dataclass
class ErrorBoundQuantizedPress(BasePress):
    error_bound: float = 0.01 # User-specified error bound
    compression_ratio: float = 0.0

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        scale = 2 * self.error_bound
        quantized_keys = torch.round(keys / scale) * scale
        quantized_values = torch.round(values / scale) * scale

        return quantized_keys, quantized_values