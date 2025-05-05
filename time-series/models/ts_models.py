import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union
from loguru import logger
import math

from time_series.models.model_registry import register_model
from time_series.models.fan_layers import (
    FANLayer, FANLayerGated, FANPhaseOffsetLayer, 
    FANPhaseOffsetLayerGated, HybridPhaseFANLayer
)




class TSForecastModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


@register_model('FANForecaster')
class FANForecaster(TSForecastModel):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        n_fan_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dim, output_dim, horizon, dropout)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.fan_layers = nn.ModuleList([
            FANLayer(hidden_dim, hidden_dim)
            for _ in range(n_fan_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = x[:, -1, :]
        
        x = self.input_proj(x)
        x = F.gelu(x)
        
        for layer in self.fan_layers:
            x = layer(x)
            x = self.dropout(x)
            
        x = self.output_proj(x)
        
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x


@register_model('FANGatedForecaster')
class FANGatedForecaster(TSForecastModel):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        n_fan_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dim, output_dim, horizon, dropout)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.fan_layers = nn.ModuleList([
            FANLayerGated(hidden_dim, hidden_dim)
            for _ in range(n_fan_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = x[:, -1, :]
        
        x = self.input_proj(x)
        x = F.gelu(x)
        
        for layer in self.fan_layers:
            x = layer(x)
            x = self.dropout(x)
            
        x = self.output_proj(x)
        
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x


@register_model('LSTMForecaster')
class LSTMForecaster(TSForecastModel):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dim, output_dim, horizon, dropout)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        outputs, (h_n, c_n) = self.lstm(x)
        
        x = h_n[-1]
        
        x = self.output_proj(x)
        
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x


@register_model('TransformerForecaster')
class TransformerForecaster(TSForecastModel):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dim, output_dim, horizon, dropout)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4*hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x = self.input_proj(x)
        
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x)
        
        x = x[:, -1, :]
        
        x = self.output_proj(x)
        
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x


@register_model('PhaseOffsetForecaster')
class PhaseOffsetForecaster(TSForecastModel):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        n_fan_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dim, output_dim, horizon, dropout)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.fan_layers = nn.ModuleList([
            FANPhaseOffsetLayer(hidden_dim, hidden_dim)
            for _ in range(n_fan_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = x[:, -1, :]
        
        x = self.input_proj(x)
        x = F.gelu(x)
        
        for layer in self.fan_layers:
            x = layer(x)
            x = self.dropout(x)
            
        x = self.output_proj(x)
        
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x


@register_model('PhaseOffsetGatedForecaster')
class PhaseOffsetGatedForecaster(TSForecastModel):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        n_fan_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dim, output_dim, horizon, dropout)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.fan_layers = nn.ModuleList([
            FANPhaseOffsetLayerGated(hidden_dim, hidden_dim)
            for _ in range(n_fan_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = x[:, -1, :]
        
        x = self.input_proj(x)
        x = F.gelu(x)
        
        for layer in self.fan_layers:
            x = layer(x)
            x = self.dropout(x)
            
        x = self.output_proj(x)
        
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x


@register_model('FANGatedTransformerForecaster')
class FANGatedTransformerForecaster(TSForecastModel):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dim, output_dim, horizon, dropout)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        self.encoder_layers = nn.ModuleList([
            FANGatedTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim * horizon)
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x = self.input_proj(x)
        
        x = self.pos_encoder(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
            
        x = self.norm(x)
        
        x = x[:, -1, :]
        
        x = self.output_proj(x)
        
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x


class FANGatedTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.fan_gated = FANLayerGated(
            input_dim=d_model,
            hidden_dim=4*d_model
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        batch_size, seq_len, d_model = src.shape
        src2 = src.view(-1, d_model)
        src2 = self.fan_gated(src2)
        src2 = src2.view(batch_size, seq_len, d_model)
        
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


@register_model('HybridPhaseFANForecaster')
class HybridPhaseFANForecaster(TSForecastModel):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        n_fan_layers: int = 3,
        dropout: float = 0.1,
        phase_decay_epochs: int = 20
    ):
        super().__init__(input_dim, hidden_dim, output_dim, horizon, dropout)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.fan_layers = nn.ModuleList([
            HybridPhaseFANLayer(hidden_dim, hidden_dim, init_phase=math.pi/4)
            for _ in range(n_fan_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, output_dim * horizon)
        
        self.phase_decay_epochs = phase_decay_epochs
        self.current_epoch = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = x[:, -1, :]
        
        x = self.input_proj(x)
        x = F.gelu(x)
        
        for layer in self.fan_layers:
            x = layer(x)
            x = self.dropout(x)
            
        x = self.output_proj(x)
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x
    
    def update_phase_coefficients(self, epoch):
        self.current_epoch = epoch
        for layer in self.fan_layers:
            layer.update_phase_coefficient(epoch, self.phase_decay_epochs)


@register_model('HybridPhaseFANTransformerForecaster')
class HybridPhaseFANTransformerForecaster(TSForecastModel):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        horizon: int = 12,
        d_model: int = 128,
        fan_dim: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_checkpointing: bool = False,
        phase_decay_epochs: int = 20
    ):
        super().__init__(input_dim, d_model, output_dim, horizon, dropout)
        
        self.hidden_dim = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.ModuleList([
            HybridPhaseTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                fan_dim=fan_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.phase_decay_epochs = phase_decay_epochs
        
        self.transformer_encoder = HybridPhaseTransformerEncoder(
            encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.transformer_encoder.use_checkpointing = use_checkpointing
        
        self.output_proj = nn.Linear(d_model, output_dim * horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x = self.input_proj(x)
        
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x)
        
        x = x[:, -1, :]
        
        x = self.output_proj(x)
        
        x = x.view(batch_size, self.horizon, self.output_dim)
        
        return x
    
    def update_phase_coefficients(self, epoch):
        self.transformer_encoder.update_phase_coefficients(epoch, self.phase_decay_epochs)


class HybridPhaseTransformerEncoder(nn.Module):
    def __init__(self, encoder_layers, norm=None):
        super().__init__()
        self.layers = encoder_layers
        self.norm = norm
        self.use_checkpointing = False
    
    def forward(self, src):
        output = src
        
        for layer in self.layers:
            if self.use_checkpointing and torch.is_grad_enabled():
                output = torch.utils.checkpoint.checkpoint(layer, output)
            else:
                output = layer(output)
                
        if self.norm is not None:
            output = self.norm(output)
            
        return output
    
    def update_phase_coefficients(self, epoch, phase_decay_epochs):
        for layer in self.layers:
            if hasattr(layer, 'update_phase_coefficient'):
                layer.update_phase_coefficient(epoch, phase_decay_epochs)


class HybridPhaseTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, fan_dim=None, dropout=0.1):
        super().__init__()
        
        if fan_dim is None:
            fan_dim = d_model * 2
            
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.hybrid_phase_fan = HybridPhaseFANLayer(
            input_dim=d_model,
            hidden_dim=fan_dim,
            init_phase=math.pi/4,
            activation="gelu"
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        batch_size, seq_len, d_model = src.shape
        src2 = src.view(-1, d_model)
        src2 = self.hybrid_phase_fan(src2)
        src2 = src2.view(batch_size, seq_len, d_model)
        
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    def update_phase_coefficient(self, epoch, phase_decay_epochs):
        self.hybrid_phase_fan.update_phase_coefficient(epoch, phase_decay_epochs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)