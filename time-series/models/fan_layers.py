import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
from loguru import logger

class HybridPhaseFANLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: Optional[int] = None, 
        num_freqs: Optional[int] = None,
        init_phase: float = math.pi/4,
        activation: str = "gelu"
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * input_dim
        
        if num_freqs is None:
            num_freqs = hidden_dim // 2
            
        if input_dim > 100:
            num_freqs = min(num_freqs, 16)
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        
        self.freqs = nn.Parameter(torch.randn(input_dim, num_freqs) * 0.01)
        self.phases = nn.Parameter(torch.ones(num_freqs) * init_phase)
        self.out_proj = nn.Linear(2 * num_freqs, input_dim)
        self.register_buffer('phase_coefficient', torch.tensor(1.0))
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.activation = F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch_size, feat_dim = x.shape
        
        if feat_dim != self.input_dim:
            if feat_dim > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padded = torch.zeros(batch_size, self.input_dim, device=x.device)
                padded[:, :feat_dim] = x
                x = padded
                
        args = torch.matmul(x, self.freqs)
        weighted_phases = self.phases * self.phase_coefficient
        args = args + weighted_phases
        
        cos_out = torch.cos(args)
        sin_out = torch.sin(args)
        
        fan_out = torch.cat([cos_out, sin_out], dim=1)
        
        out = self.out_proj(fan_out)
        out = self.activation(out)
            
        return out
    
    def update_phase_coefficient(self, epoch, total_phase_decay_epochs):
        if epoch < total_phase_decay_epochs:
            coefficient = 1.0 - (epoch / total_phase_decay_epochs)
            self.phase_coefficient = torch.tensor(coefficient, device=self.phase_coefficient.device)
        else:
            self.phase_coefficient = torch.tensor(0.0, device=self.phase_coefficient.device)

class FANLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: Optional[int] = None, 
        num_freqs: Optional[int] = None,
        activation: str = "gelu"
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * input_dim
        
        if num_freqs is None:
            num_freqs = hidden_dim // 2
        
        if input_dim > 100:
            num_freqs = min(num_freqs, 16)
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        
        self.freqs = nn.Parameter(torch.randn(input_dim, num_freqs) * 0.01)
        
        self.phases = nn.Parameter(torch.randn(input_dim, num_freqs) * 0.01)
        
        self.out_proj = nn.Linear(2 * num_freqs, input_dim)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.activation = F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch_size, feat_dim = x.shape
        
        if feat_dim != self.input_dim:
            logger.warning(f"Input feature dimension {feat_dim} doesn't match expected {self.input_dim}")
            if feat_dim > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padded = torch.zeros(batch_size, self.input_dim, device=x.device)
                padded[:, :feat_dim] = x
                x = padded
                
        args = torch.matmul(x, self.freqs)
        
        phase_sums = torch.sum(self.phases, dim=0)
        
        args = args + phase_sums
        
        cos_out = torch.cos(args)
        sin_out = torch.sin(args)
        
        fan_out = torch.cat([cos_out, sin_out], dim=1)
        
        out = self.out_proj(fan_out)
        out = self.activation(out)
            
        return out


class FANLayerGated(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: Optional[int] = None, 
        num_freqs: Optional[int] = None,
        activation: str = "gelu"
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * input_dim
        
        if num_freqs is None:
            num_freqs = hidden_dim // 2
            
        if input_dim > 100:
            num_freqs = min(num_freqs, 16)
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        
        self.freqs = nn.Parameter(torch.randn(input_dim, num_freqs) * 0.01)
        
        self.phases = nn.Parameter(torch.randn(input_dim, num_freqs) * 0.01)
        
        self.gates = nn.Parameter(torch.ones(num_freqs * 2))
        
        self.out_proj = nn.Linear(2 * num_freqs, input_dim)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.activation = F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch_size, feat_dim = x.shape
        
        if feat_dim != self.input_dim:
            logger.warning(f"Input feature dimension {feat_dim} doesn't match expected {self.input_dim}")
            if feat_dim > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padded = torch.zeros(batch_size, self.input_dim, device=x.device)
                padded[:, :feat_dim] = x
                x = padded
                
        args = torch.matmul(x, self.freqs)
        
        phase_sums = torch.sum(self.phases, dim=0)
        
        args = args + phase_sums
        
        cos_out = torch.cos(args)
        sin_out = torch.sin(args)
        
        gates = torch.sigmoid(self.gates)
        cos_gates = gates[:self.num_freqs]
        sin_gates = gates[self.num_freqs:]
        
        cos_out = cos_out * cos_gates
        sin_out = sin_out * sin_gates
        
        fan_out = torch.cat([cos_out, sin_out], dim=1)
        
        out = self.out_proj(fan_out)
        out = self.activation(out)
            
        return out


class FANPhaseOffsetLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: Optional[int] = None, 
        num_freqs: Optional[int] = None,
        activation: str = "gelu"
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * input_dim
        
        if num_freqs is None:
            num_freqs = hidden_dim // 2
            
        if input_dim > 100:
            num_freqs = min(num_freqs, 16)
            
        p_dim = hidden_dim // 4
        g_dim = hidden_dim // 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.freqs = nn.Parameter(torch.randn(input_dim, p_dim) * 0.01)
        
        self.offset = nn.Parameter(torch.full((p_dim,), math.pi / 4))
        
        self.input_linear_g = nn.Linear(input_dim, g_dim)
        
        self.out_proj = nn.Linear(2 * p_dim + g_dim, input_dim)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.activation = F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch_size, feat_dim = x.shape
        
        if feat_dim != self.input_dim:
            logger.warning(f"Input feature dimension {feat_dim} doesn't match expected {self.input_dim}")
            if feat_dim > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padded = torch.zeros(batch_size, self.input_dim, device=x.device)
                padded[:, :feat_dim] = x
                x = padded
        
        p = torch.matmul(x, self.freqs)
        
        g = self.activation(self.input_linear_g(x))
        
        p_plus = p + self.offset
        p_minus = p + (torch.pi / 2) - self.offset
        
        sin_out = torch.sin(p_plus)
        cos_out = torch.cos(p_minus)
        
        fan_out = torch.cat([cos_out, sin_out, g], dim=1)
        
        out = self.out_proj(fan_out)
        out = self.activation(out)
            
        return out


class FANPhaseOffsetLayerGated(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: Optional[int] = None, 
        num_freqs: Optional[int] = None,
        activation: str = "gelu"
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * input_dim
        
        if num_freqs is None:
            num_freqs = hidden_dim // 2
            
        if input_dim > 100:
            num_freqs = min(num_freqs, 16)
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        
        self.freqs = nn.Parameter(torch.randn(input_dim, num_freqs) * 0.01)
        
        self.offset = nn.Parameter(torch.full((num_freqs,), math.pi / 4))
        
        self.gates = nn.Parameter(torch.ones(num_freqs * 2))
        
        self.out_proj = nn.Linear(2 * num_freqs, input_dim)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.activation = F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch_size, feat_dim = x.shape
        
        if feat_dim != self.input_dim:
            logger.warning(f"Input feature dimension {feat_dim} doesn't match expected {self.input_dim}")
            if feat_dim > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padded = torch.zeros(batch_size, self.input_dim, device=x.device)
                padded[:, :feat_dim] = x
                x = padded
        
        args = torch.matmul(x, self.freqs)
        
        p_plus = args + self.offset
        p_minus = args + (torch.pi / 2) - self.offset
        
        sin_out = torch.sin(p_plus)
        cos_out = torch.cos(p_minus)
        
        gates = torch.sigmoid(self.gates)
        cos_gates = gates[:self.num_freqs]
        sin_gates = gates[self.num_freqs:]
        
        cos_out = cos_out * cos_gates
        sin_out = sin_out * sin_gates
        
        fan_out = torch.cat([cos_out, sin_out], dim=1)
        
        out = self.out_proj(fan_out)
        out = self.activation(out)
            
        return out