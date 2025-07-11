import math
from dataclasses import dataclass
import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class Vayne1Press(BasePress):
    """
    Vayne 1 is a second version.  
    It includes a few key features:  
    (* denotes for added features)
        1. The key tensor is quantized channel-wise (excluding the first token), and the value tensor is quantized token-wise.
        2. A de-RoPE transformation is applied to recover the key tensor before RoPE.
        3. The first row of the key tensor is preserved in full precision due to its large data range.
        4. * It includes a new block-wise fixed length encoding. (only key tensor, channel-wise)
    """
    compression_ratio: float = 0.0  # dummy field to satisfy kvpress interface
    nbits: int = 3
    total_original_bits: int = 0
    total_compressed_bits: int = 0

    def compute_compression_ratio_channelwise(self, quantized_tensor, block_size=16):
        rows, cols = quantized_tensor.shape
        qbits = self.nbits

        pad_rows = (block_size - (rows % block_size)) % block_size
        if pad_rows > 0:
            pad_tensor = torch.zeros((pad_rows, cols), dtype=quantized_tensor.dtype, device=quantized_tensor.device)
            quantized_tensor = torch.cat([quantized_tensor, pad_tensor], dim=0)

        total_rows = quantized_tensor.shape[0]
        num_blocks = total_rows // block_size
        reshaped = quantized_tensor.reshape(num_blocks, block_size, cols)  # [B, S, C]

        block_min = reshaped.min(dim=1).values
        block_max = reshaped.max(dim=1).values
        diff = (block_max - block_min).clamp(min=1)

        bits_per_val = torch.ceil(torch.log2(diff.float() + 1)).to(torch.int32)
        metadata_bits = qbits + math.ceil(math.log2(qbits + 1))
        compressed_bits = metadata_bits + block_size * bits_per_val

        total_compressed = compressed_bits.sum().item()
        total_original = block_size * qbits * num_blocks * cols

        return total_original, total_compressed

    def compute_compression_ratio_tokenwise(self, quantized_tensor, block_size=16):
        rows, cols = quantized_tensor.shape
        qbits = self.nbits

        pad_cols = (block_size - (cols % block_size)) % block_size
        if pad_cols > 0:
            pad_tensor = torch.zeros((rows, pad_cols), dtype=quantized_tensor.dtype, device=quantized_tensor.device)
            quantized_tensor = torch.cat([quantized_tensor, pad_tensor], dim=1)

        total_cols = quantized_tensor.shape[1]
        num_blocks = total_cols // block_size
        reshaped = quantized_tensor.reshape(rows, num_blocks, block_size)  # [T, B, S]

        block_min = reshaped.min(dim=2).values
        block_max = reshaped.max(dim=2).values
        diff = (block_max - block_min).clamp(min=1)

        bits_per_val = torch.ceil(torch.log2(diff.float() + 1)).to(torch.int32)
        metadata_bits = qbits + math.ceil(math.log2(qbits + 1))
        compressed_bits = metadata_bits + block_size * bits_per_val

        total_compressed = compressed_bits.sum().item()
        total_original = block_size * qbits * rows * num_blocks

        return total_original, total_compressed

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
        keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * (-sin.unsqueeze(1)))

        key_first_token = keys[:, :, 0:1, :]  # [B, H, 1, D]
        key_rest = keys[:, :, 1:, :]  # [B, H, T-1, D]
        B, H, Tm1, D = key_rest.shape
        key_flat = key_rest.reshape(-1, Tm1, D)

        key_min = key_flat.min(dim=1, keepdim=True)[0]
        key_max = key_flat.max(dim=1, keepdim=True)[0]
        scale = (key_max - key_min) / qmax
        scale = torch.clamp(scale, min=1e-8)

        key_q = ((key_flat - key_min) / scale).round().clamp(0, qmax)

        for i in range(key_q.shape[0]):
            orig, comp = self.compute_compression_ratio_channelwise(key_q[i])
            self.total_original_bits += orig
            self.total_compressed_bits += comp

        key_deq = key_q * scale + key_min
        key_rest = key_deq.reshape(B, H, Tm1, D)
        keys = torch.cat([key_first_token, key_rest], dim=2)
        keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * sin.unsqueeze(1))

        value_flat = values.reshape(-1, values.shape[2], values.shape[3])
        val_min = value_flat.min(dim=2, keepdim=True)[0]
        val_max = value_flat.max(dim=2, keepdim=True)[0]
        val_scale = (val_max - val_min) / qmax
        val_scale = torch.clamp(val_scale, min=1e-8)

        val_q = ((value_flat - val_min) / val_scale).round().clamp(0, qmax)

        # for i in range(val_q.shape[0]):
        #     orig, comp = self.compute_compression_ratio_tokenwise(val_q[i])
        #     self.total_original_bits += orig
        #     self.total_compressed_bits += comp

        val_deq = val_q * val_scale + val_min
        values = val_deq.reshape_as(values)

        return keys, values