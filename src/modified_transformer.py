import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from loguru import logger

from src.models import register_model
from src.attention import AttentionLayer, FullAttention, ProbAttention


class ModifiedEncoderLayer(nn.Module):
    """
    Modified Transformer encoder layer with enhanced attention mechanism.
    This encoder combines standard attention with probabilistic attention
    for better handling of long sequences.
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(ModifiedEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # Standard self-attention 
        self.attention1 = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads)
            
        # Probabilistic attention for handling sparse dependencies
        self.attention2 = AttentionLayer(
            ProbAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable gate parameter for balancing the two attention mechanisms
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, attn_mask=None):
        # Standard attention branch
        attn1, _ = self.attention1(x, x, x, attn_mask=attn_mask)
        
        # Probabilistic attention branch
        attn2, _ = self.attention2(x, x, x, attn_mask=attn_mask)
        
        # Gated combination of attention mechanisms
        gate_val = torch.sigmoid(self.gate)
        combined_attn = gate_val * attn1 + (1 - gate_val) * attn2
        
        # Add & Norm
        x = x + self.dropout(combined_attn)
        x = self.norm1(x)
        
        # Feed Forward
        ff = self.feed_forward(x)
        x = x + self.dropout(ff)
        x = self.norm2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Improved positional encoding with learnable weights
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (non-learnable) but also create a learnable scaling factor
        self.register_buffer('pe', pe)
        self.weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Apply positional encoding with learnable weight
        x = x + self.weight * self.pe[:, :x.size(1), :]
        return self.dropout(x)


@register_model("ModifiedTransformer")
class ModifiedTransformer(nn.Module):
    """
    Modified Transformer with enhanced architecture:
    - Dual attention mechanism (standard + probabilistic)
    - Learnable positional encoding
    - Layer-specific gating
    """
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        hidden_dim=512,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        ff_dim=None,
        activation="gelu",
        max_seq_len=1024
    ):
        super(ModifiedTransformer, self).__init__()
        logger.info("Initializing ModifiedTransformer model...")
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Enhanced positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Modified encoder layers
        self.encoder_layers = nn.ModuleList([
            ModifiedEncoderLayer(
                d_model=hidden_dim,
                n_heads=num_heads,
                d_ff=ff_dim,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        # Layer-specific adaptive weights
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, attn_mask=None):
        # Input embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Store layer outputs for weighted combination
        layer_outputs = []
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)
            layer_outputs.append(x)
            
        # Apply softmax to ensure weights sum to 1
        norm_weights = F.softmax(self.layer_weights, dim=0)
        
        # Weighted combination of layer outputs
        x = sum(w * output for w, output in zip(norm_weights, layer_outputs))
        
        # Output projection
        return self.output_layer(x)


@register_model("ModifiedTransformerForecaster")
class ModifiedTransformerForecaster(nn.Module):
    """
    Modified transformer adapted for time series forecasting with enhanced features.
    """
    def __init__(
        self,
        input_dim=1,
        seq_len=96,
        pred_len=24,
        hidden_dim=512,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
        ff_dim=None,
        activation="gelu"
    ):
        super(ModifiedTransformerForecaster, self).__init__()
        logger.info("Initializing ModifiedTransformerForecaster model...")
        
        # Configuration
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # Time series specific embedding with frequency encoding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Modified transformer backbone
        self.transformer = ModifiedTransformer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ff_dim=ff_dim,
            activation=activation,
            max_seq_len=seq_len
        )
        
        # Output projection with additional capacity for forecasting
        self.pre_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(hidden_dim, pred_len)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len*input_dim]
        batch_size = x.shape[0]
        
        # Handle input shape mismatches
        expected_elements = self.seq_len * self.input_dim
        actual_elements = x.shape[1]
        
        if actual_elements != expected_elements:
            logger.warning(f"Input shape mismatch: got {actual_elements} elements, expected {expected_elements}")
            
            if actual_elements > expected_elements:
                # Truncate if input has more elements than expected
                logger.warning(f"Truncating input from {actual_elements} to {expected_elements} elements")
                x = x[:, :expected_elements]
            else:
                # Pad with zeros if input has fewer elements than expected
                logger.warning(f"Padding input from {actual_elements} to {expected_elements} elements")
                padding = torch.zeros(batch_size, expected_elements - actual_elements, device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        # Reshape to [batch_size, seq_len, input_dim]
        x = x.view(batch_size, self.seq_len, self.input_dim)
        
        # Apply embedding 
        x = self.embedding(x)
        
        # Apply transformer backbone
        x = self.transformer(x)
        
        # Extract features from the last token
        x = x[:, -1, :]  # [batch_size, hidden_dim]
        
        # Apply pre-output processing and final projection
        x = self.pre_output(x)
        out = self.output_layer(x)  # [batch_size, pred_len]
        
        return out