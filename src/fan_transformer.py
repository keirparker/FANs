import torch
import torch.nn as nn
from loguru import logger
import math

from src.models import register_model
from src.attention import AttentionLayer, FullAttention


class FANTransformerLayer(nn.Module):
    """
    A transformer layer with FAN-like processing.
    Combines self-attention with FAN-style trigonometric encoding.
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, attention_dropout=0.1, bias=True, gated=True):
        super(FANTransformerLayer, self).__init__()
        
        # 1. Dimensions for FAN components
        p_dim = hidden_dim // 4  # For trigonometric functions 
        g_dim = hidden_dim // 2  # For nonlinear branch
        
        # 2. Self-attention component
        self.attention = AttentionLayer(
            attention=FullAttention(attention_dropout=attention_dropout),
            d_model=hidden_dim,
            n_heads=num_heads
        )
        
        # 3. FAN-specific components
        self.input_linear_p = nn.Linear(hidden_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(hidden_dim, g_dim, bias=bias)
        self.activation = nn.GELU()
        
        # 4. Per-feature phase offset and gate parameter (if gated=True)
        self.offset = nn.Parameter(torch.full((p_dim,), math.pi / 4))
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
        
        # 5. Layer norm and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        # 1. Apply self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # 2. Apply FAN-like processing
        identity = x
        
        # Compute projections
        p = self.input_linear_p(x)  # Shape: (..., p_dim)
        g = self.activation(self.input_linear_g(x))  # Shape: (..., g_dim)
        
        # Compute adjusted phases
        p_plus = p + self.offset  # For sine
        p_minus = p + (torch.pi / 2) - self.offset  # For cosine
        
        # Trigonometric branch
        sin_part = torch.sin(p_plus)
        cos_part = torch.cos(p_minus)
        
        if self.gated:
            # Compute gate value γ = sigmoid(s)
            gamma = torch.sigmoid(self.gate)
            
            # Apply gating
            fourier_branch = gamma * torch.cat((sin_part, cos_part), dim=-1)
            nonlinear_branch = (1 - gamma) * g
            
            # Concatenate branches to form output
            fan_output = torch.cat((fourier_branch, nonlinear_branch), dim=-1)
        else:
            # No gating - standard concatenation
            fan_output = torch.cat((sin_part, cos_part, g), dim=-1)
        
        # 3. Add residual connection and layer norm
        x = identity + self.dropout(fan_output)
        x = self.norm2(x)
        
        return x


@register_model("FANTransformer")
class FANTransformer(nn.Module):
    """
    Transformer model that incorporates FAN-style processing in each layer.
    
    Structure:
    1) Input embedding with positional encoding
    2) Multiple FANTransformerLayer layers
    3) Output projection
    """
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        hidden_dim=512,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        max_seq_len=1024,
        bias=True,
        gated=True
    ):
        super(FANTransformer, self).__init__()
        logger.info(f"Initializing FANTransformer model (gated={gated})...")
        
        # 1. Input embedding with positional encoding
        self.input_embedding = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.pos_encoding = self._create_positional_encoding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Stack of FANTransformerLayer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                FANTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    bias=bias,
                    gated=gated
                )
            )
        
        # 3. Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=bias)
        
    def _create_positional_encoding(self, max_seq_len, hidden_dim):
        """Create fixed sinusoidal positional embeddings"""
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim))
        
        pe = torch.zeros(1, max_seq_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe, requires_grad=False)
    
    def forward(self, x, attn_mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Apply input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Pass through each transformer layer
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        # Apply output projection (can be customized based on task)
        # For sequence-to-sequence: apply to all positions
        # For classification/regression: take final position or perform pooling
        output = self.output_projection(x)
        
        return output


@register_model("FANTransformerForecaster")
@register_model("FANGatedTransformerForecaster")  # Register the gated variant with its own name
class FANTransformerForecaster(nn.Module):
    """
    FANTransformer adapted for time series forecasting.
    
    This model takes a sequence of observations and predicts the next pred_len values.
    """
    def __init__(
        self,
        input_dim=1,
        seq_len=96,
        pred_len=24,
        hidden_dim=512,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        bias=True,
        gated=True
    ):
        super(FANTransformerForecaster, self).__init__()
        logger.info(f"Initializing FANTransformerForecaster model (gated={gated})...")
        
        # Configuration
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        
        # Time series specific embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.xavier_uniform_(self.pos_encoding)
        
        # FANTransformer backbone
        self.transformer = FANTransformer(
            input_dim=hidden_dim,  # We've already embedded the input
            output_dim=hidden_dim,  # Keep in hidden dimension for now
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            max_seq_len=seq_len,
            bias=bias,
            gated=gated
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
        
        # Apply embedding
        x = self.embedding(x) + self.pos_encoding
        
        # Apply transformer backbone
        x = self.transformer(x)
        
        # Take features from the last token for forecasting
        x = x[:, -1, :]  # [batch_size, hidden_dim]
        
        # Project to prediction horizon
        out = self.output_layer(x)  # [batch_size, pred_len]
        
        return out
        
