import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from loguru import logger

from time_series.models.model_registry import register_model
from time_series.models.fan_layers import FANPhaseOffsetLayer, FANPhaseOffsetLayerGated


class PhaseOffsetTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        fan_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.fan = FANPhaseOffsetLayer(d_model, fan_dim)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src2, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        batch_size, seq_len, d_model = src.shape
        
        self._layer_idx = getattr(self, '_layer_idx', 0)
        
        src_flat = src.reshape(-1, d_model)
        
        src2_flat = self.fan(src_flat)
        
        src2 = src2_flat.reshape(batch_size, seq_len, d_model)
        
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


@register_model('PhaseOffsetTransformerForecaster')
class PhaseOffsetTransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        nhead: int = 8,
        fan_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_checkpointing: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.horizon = horizon
        self.use_checkpointing = use_checkpointing
        
        self.training_iterations = 0
        self.epoch_times = []
        self.train_loss_history = []
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = PhaseOffsetTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                fan_dim=fan_dim,
                dropout=dropout
            )
            layer._layer_idx = i
            self.encoder_layers.append(layer)
        
        self.output_proj = nn.Linear(d_model, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.training_iterations += 1
        batch_size = x.shape[0]
        
        x = self.input_proj(x)
        
        x = self.pos_encoder(x)
        
        for i, layer in enumerate(self.encoder_layers):
            if self.use_checkpointing and self.training and hasattr(torch, 'utils') and hasattr(torch.utils, 'checkpoint'):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(inputs[0])
                    return custom_forward
                
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    use_reentrant=False
                )
            else:
                x = layer(x)
        
        x = x[:, -1, :]
        
        x = self.output_proj(x)
        
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.forward_times = []

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            import time
            start_time = time.time()
            
        x = x + self.pe[:, :x.size(1), :]
        result = self.dropout(x)
        
        if self.training:
            self.forward_times.append(time.time() - start_time)
            
        return result