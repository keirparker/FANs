import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from loguru import logger

from src.models import register_model
from src.attention import AttentionLayer, FullAttention


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for transformer models
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class EncoderLayer(nn.Module):
    """
    Standard transformer encoder layer: self-attention + feed-forward
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Self Attention
        x_attn, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # Add & Norm
        x = x + self.dropout(x_attn)
        x = self.norm1(x)

        # Feed Forward
        y = self.feed_forward(x)
        x = x + self.dropout(y)
        x = self.norm2(x)

        return x


class PositionwiseFeedForward(nn.Module):
    """
    Two-layer MLP for transformer feed-forward blocks
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


@register_model("StandardTransformer")
class StandardTransformer(nn.Module):
    """
    Standard transformer encoder model
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
        super(StandardTransformer, self).__init__()
        logger.info("Initializing StandardTransformer model...")
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=hidden_dim,
                n_heads=num_heads,
                d_ff=ff_dim,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        # Input embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer encoder blocks
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)
            
        # Output projection
        output = self.output_layer(x)
        
        return output


@register_model("StandardTransformerForecaster") 
class StandardTransformerForecaster(nn.Module):
    """
    Standard transformer adapted for time series forecasting.
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
        super(StandardTransformerForecaster, self).__init__()
        logger.info("Initializing StandardTransformerForecaster model...")
        
        # Configuration
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        
        # Time series specific embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, seq_len)
        
        # Transformer encoder
        self.transformer = StandardTransformer(
            input_dim=hidden_dim,  # We've already embedded the input
            output_dim=hidden_dim,  # Keep in hidden dimension for projection
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ff_dim=ff_dim,
            activation=activation,
            max_seq_len=seq_len
        )
        
        # Output projection for forecasting
        self.output_layer = nn.Linear(hidden_dim, pred_len)
        
        
    def forward(self, x):
        # x shape: [batch_size, seq_len*input_dim]
        batch_size = x.shape[0]
        
        # Check if the input tensor has the expected number of elements
        expected_elements = self.seq_len * self.input_dim
        actual_elements = x.shape[1]
        
        if actual_elements != expected_elements:
            logger.warning(f"Input shape mismatch: got {actual_elements} elements, expected {expected_elements}")
            
            # Fix by reshaping input to match model's expectations
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
        
        # Apply embedding and positional encoding
        x = self.pos_encoder(self.embedding(x))
        
        # Apply transformer backbone
        x = self.transformer(x)
        
        # Take features from the last token for forecasting
        x = x[:, -1, :]  # [batch_size, hidden_dim]
        
        # Project to prediction horizon
        out = self.output_layer(x)  # [batch_size, pred_len]
        
        return out
        
