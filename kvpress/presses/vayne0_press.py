from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class Vayne0Press(BasePress):
    """
    Vayne 0 is a prototype for super-fast KV compression.  
    It includes a few key features:  
        1. The key tensor is quantized channel-wise (excluding the first token), and the value tensor is quantized token-wise.
        2. A de-RoPE transformation is applied to recover the key tensor before RoPE.
        3. The first row of the key tensor is preserved in full precision due to its large data range.
    """

    compression_ratio: float = 0.0
    nbits: int = 3

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        abits = self.nbits
        qmax = 2 ** abits - 1

        cos, sin = kwargs["position_embeddings"]
        # De-apply RoPE
        keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * (-sin.unsqueeze(1)))

        # === Key: Channel-wise quantization (skip token 0) ===
        key_first_token = keys[:, :, 0:1, :]  # [B, H, 1, D]
        key_rest = keys[:, :, 1:, :]  # [B, H, T-1, D]

        # Flatten to [B*H, T-1, D] for convenience
        B, H, Tm1, D = key_rest.shape[0], key_rest.shape[1], key_rest.shape[2], key_rest.shape[3]
        key_flat = key_rest.reshape(-1, Tm1, D)

        # Channel-wise (per D) quantization
        key_min = key_flat.min(dim=1, keepdim=True)[0]
        key_max = key_flat.max(dim=1, keepdim=True)[0]
        scale = (key_max - key_min) / qmax
        scale = torch.clamp(scale, min=1e-8)

        key_q = ((key_flat - key_min) / scale).round().clamp(0, qmax)
        key_deq = key_q * scale + key_min
        key_rest = key_deq.reshape(B, H, Tm1, D)

        keys = torch.cat([key_first_token, key_rest], dim=2)

        # Re-apply RoPE
        keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * sin.unsqueeze(1))

        # === Value: Token-wise quantization ===
        # [B, H, T, D] -> [B*H, T, D]
        value_flat = values.reshape(-1, values.shape[2], values.shape[3])
        val_min = value_flat.min(dim=2, keepdim=True)[0]
        val_max = value_flat.max(dim=2, keepdim=True)[0]
        val_scale = (val_max - val_min) / qmax
        val_scale = torch.clamp(val_scale, min=1e-8)

        val_q = ((value_flat - val_min) / val_scale).round().clamp(0, qmax)
        val_deq = val_q * val_scale + val_min
        values = val_deq.reshape_as(values)

        return keys, values