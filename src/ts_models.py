#!/usr/bin/env python
"""
Time series forecasting models based on FAN architecture.
These models are designed to handle sequence data for time series forecasting.

Author: GitHub Copilot for keirparker
Last updated: 2025-03-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.models import register_model


class TimeSeriesEmbedding(nn.Module):
    """
    Embedding layer for time series data with support for position encoding.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int):
        super(TimeSeriesEmbedding, self).__init__()
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        # Optional positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.xavier_uniform_(self.pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Apply feature embedding (to each timestep)
        x = x.view(batch_size * seq_len, -1)  # [batch_size*seq_len, input_dim]
        x = self.feature_embedding(x)  # [batch_size*seq_len, hidden_dim]
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_dim]

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        return x


@register_model("FANForecaster")
class FANForecaster(nn.Module):
    """
    FAN model adapted for time series forecasting.
    Uses FANLayer for processing sequential data.
    """

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 96,
        pred_len: int = 24,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super(FANForecaster, self).__init__()
        logger.info("Initializing FANForecaster model...")

        # Configuration
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Time series specific embedding
        self.embedding = TimeSeriesEmbedding(input_dim, hidden_dim, seq_len)

        # Sequence of FAN layers
        self.layers = nn.ModuleList()
        from src.models import FANLayer

        for _ in range(num_layers - 1):
            self.layers.append(FANLayer(hidden_dim, hidden_dim))

        # Projection for sequence to prediction
        self.seq_projection = nn.Linear(seq_len * hidden_dim, hidden_dim)

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]

        # Apply FAN layers
        for layer in self.layers:
            x = layer(x)  # [batch_size, seq_len, hidden_dim]

        # Flatten sequence dimension
        x = x.view(batch_size, -1)  # [batch_size, seq_len*hidden_dim]

        # Project to hidden dimension
        x = F.relu(self.seq_projection(x))  # [batch_size, hidden_dim]

        # Final output projection to prediction horizon
        out = self.output_layer(x)  # [batch_size, pred_len]

        return out


@register_model("FANGatedForecaster")
class FANGatedForecaster(nn.Module):
    """
    FANGated model adapted for time series forecasting.
    Uses FANLayerGated for processing sequential data.
    """

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 96,
        pred_len: int = 24,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super(FANGatedForecaster, self).__init__()
        logger.info("Initializing FANGatedForecaster model...")

        # Configuration
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Time series specific embedding
        self.embedding = TimeSeriesEmbedding(input_dim, hidden_dim, seq_len)

        # Sequence of FANGated layers
        self.layers = nn.ModuleList()
        from src.models import FANLayerGated

        for _ in range(num_layers - 1):
            self.layers.append(FANLayerGated(hidden_dim, hidden_dim, gated=True))

        # Projection for sequence to prediction
        self.seq_projection = nn.Linear(seq_len * hidden_dim, hidden_dim)

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]

        # Apply FAN layers
        for layer in self.layers:
            x = layer(x)  # [batch_size, seq_len, hidden_dim]

        # Flatten sequence dimension
        x = x.view(batch_size, -1)  # [batch_size, seq_len*hidden_dim]

        # Project to hidden dimension
        x = F.relu(self.seq_projection(x))  # [batch_size, hidden_dim]

        # Final output projection to prediction horizon
        out = self.output_layer(x)  # [batch_size, pred_len]

        return out


@register_model("LSTM_Forecaster")
class LSTMForecaster(nn.Module):
    """
    LSTM-based model for time series forecasting.
    """

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 96,
        pred_len: int = 24,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(LSTMForecaster, self).__init__()
        logger.info("Initializing LSTM_Forecaster model...")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Feature embedding
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Apply feature embedding
        x = self.feature_embedding(x)  # [batch_size, seq_len, hidden_dim]

        # LSTM processing
        outputs, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]

        # Take final state
        final_state = outputs[:, -1, :]  # [batch_size, hidden_dim]

        # Project to prediction horizon
        out = self.output_layer(final_state)  # [batch_size, pred_len]

        return out


@register_model("Transformer_Forecaster")
class TransformerForecaster(nn.Module):
    """
    Transformer-based model for time series forecasting.
    """

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 96,
        pred_len: int = 24,
        hidden_dim: int = 512,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super(TransformerForecaster, self).__init__()
        logger.info("Initializing Transformer_Forecaster model...")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Time series embedding with position encoding
        self.embedding = TimeSeriesEmbedding(input_dim, hidden_dim, seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Apply embedding with positional encoding
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]

        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch_size, seq_len, hidden_dim]

        # Take features from the last token
        x = x[:, -1, :]  # [batch_size, hidden_dim]

        # Project to prediction horizon
        out = self.output_layer(x)  # [batch_size, pred_len]

        return out
