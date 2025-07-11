from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

@dataclass
class KiviPress(BasePress):
    nbits: int = 4
    group_size: int = 32
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

        def quantize_except_last_group(tensor, group_dim, group_size, nbits):
            qmin, qmax = 0, 2**nbits - 1
            shape = tensor.shape
            total_size = shape[group_dim]
            num_total_groups = (total_size + group_size - 1) // group_size

            # Always reserve last group as residual
            num_quant_groups = max(num_total_groups - 1, 0)
            quant_end = num_quant_groups * group_size
            residual_start = quant_end

            # Slice quantized portion
            slices_quant = [slice(None)] * len(shape)
            slices_quant[group_dim] = slice(0, quant_end)
            quant_part = tensor[tuple(slices_quant)]

            # Slice residual portion
            slices_residual = [slice(None)] * len(shape)
            slices_residual[group_dim] = slice(residual_start, total_size)
            residual_part = tensor[tuple(slices_residual)]

            if quant_part.numel() == 0:
                return residual_part

            # Reshape quant part into groups
            new_shape = list(quant_part.shape)
            new_shape[group_dim:group_dim+1] = [num_quant_groups, group_size]
            grouped = quant_part.reshape(*new_shape)

            # Per-group quantization
            min_val = grouped.amin(dim=group_dim+1, keepdim=True)
            max_val = grouped.amax(dim=group_dim+1, keepdim=True)
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero_point = qmin - min_val / scale

            q = torch.clamp(torch.round(grouped / scale + zero_point), qmin, qmax)
            dq = (q - zero_point) * scale

            dq_reshaped = dq.reshape(*quant_part.shape)

            # Concatenate quantized + residual
            output = torch.cat([dq_reshaped, residual_part], dim=group_dim)
            return output

        # Key: group along token dimension (dim=2)
        quantized_keys = quantize_except_last_group(
            keys, group_dim=2, group_size=self.group_size, nbits=self.nbits
        )

        # Value: group along head dimension (dim=3)
        quantized_values = quantize_except_last_group(
            values, group_dim=3, group_size=self.group_size, nbits=self.nbits
        )

        return quantized_keys, quantized_values