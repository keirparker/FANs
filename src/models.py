# src/src.py
from loguru import logger
import torch
import torch.nn as nn
import math

def _ensure_compatible_input(src, in_features, logger):
    """
    Utility function to ensure input tensor has the right shape for model layers.
    
    Args:
        src: Input tensor, can be 2D (batch_size, features), 3D (batch_size, seq_len, features), etc.
        in_features: Expected number of input features
        logger: Logger instance for debug output
        
    Returns:
        Reshaped tensor with compatible dimensions (batch_size, in_features)
    """
    # Store original shape for debugging
    original_shape = src.shape
    logger.debug(f"Input shape before processing: {original_shape}, in_features expected: {in_features}")
    
    # Handle special case: if src shape is [batch_size, 1] and we need more than 1 feature
    if src.dim() == 2 and src.shape[1] == 1 and in_features > 1:
        logger.debug(f"Expanding single value input from {src.shape} to match required features {in_features}")
        # This is likely a simple scalar input that needs to be expanded
        return src.expand(-1, in_features)
    
    # Handle 3D inputs (batch_size, seq_len, features) 
    if src.dim() == 3:
        # For 3D inputs, we need to flatten to 2D first
        batch_size = src.shape[0]
        logger.debug(f"Flattening 3D input with shape {src.shape}")
        src = src.reshape(batch_size, -1)
        logger.debug(f"Shape after flattening: {src.shape}")
    elif src.dim() > 3:
        # Higher dimensions not supported
        logger.warning(f"Unexpected input shape: {src.shape}, expected 2D or 3D tensor")
        # Flatten all dimensions except batch
        src = src.reshape(src.shape[0], -1)
    
    # At this point src should be 2D (batch_size, features)
    # CRITICAL FIX: ALWAYS force the tensor to have exactly in_features
    if src.shape[1] != in_features:
        if src.shape[1] > in_features:
            # If we have too many features, truncate
            logger.warning(f"Input has too many features ({src.shape[1]}) for model ({in_features}). Truncating. Original shape: {original_shape}")
            src = src[:, :in_features]
        else:
            # If we have too few features, pad with zeros
            logger.warning(f"Input has too few features ({src.shape[1]}) for model ({in_features}). Padding. Original shape: {original_shape}")
            pad = torch.zeros((src.shape[0], in_features - src.shape[1]), device=src.device)
            src = torch.cat([src, pad], dim=1)
    
    # Final sanity check
    assert src.shape[1] == in_features, f"Shape mismatch! Got {src.shape}, expected ({src.shape[0]}, {in_features})"
    logger.debug(f"Final compatible input shape: {src.shape}")
    
    return src


# Global model registry
model_registry = {}

def register_model(model_name):
    """
    Decorator to register a model class.
    """
    def decorator(cls):
        model_registry[model_name] = cls
        logger.debug(f"Registered model class: {model_name}")
        return cls
    return decorator

def get_model_by_name(model_name, *args, **kwargs):
    """
    Retrieve and instantiate a model class by its name.
    """
    # Make sure all model modules are imported
    try:
        from src.ts_models import FANForecaster, FANGatedForecaster, LSTMForecaster, TransformerForecaster
        from src.fan_transformer import FANTransformer, FANTransformerForecaster, FANGatedTransformerForecaster
        from src.transformer import StandardTransformer, StandardTransformerForecaster
        from src.modified_transformer import ModifiedTransformer, ModifiedTransformerForecaster
        from src.phase_offset_transformer import (
            FANPhaseOffsetTransformer, 
            FANPhaseOffsetTransformerForecaster,
            FANPhaseOffsetGatedTransformerForecaster
        )
        logger.debug(f"Imported all model modules. Registry has {len(model_registry)} models.")
    except ImportError as e:
        logger.warning(f"Could not import some model modules: {e}")
    
    # Get the model class from the registry
    model_cls = model_registry.get(model_name)
    if model_cls is None:
        # Log available models for debugging
        available_models = sorted(list(model_registry.keys()))
        logger.error(f"No model found with model_name '{model_name}'. Available models: {available_models}")
        raise ValueError(f"No model found with model_name {model_name}.")
        
    logger.info(f"Instantiating model: {model_name} with args={args}, kwargs={kwargs}")
    return model_cls(*args, **kwargs)

@register_model("FANLayer")
class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayer, self).__init__()
        logger.debug("Initializing FANLayer ...")
        self.input_linear_p = nn.Linear(input_dim, output_dim // 4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim - output_dim // 2))
        self.activation = nn.GELU()

    def forward(self, src):
        logger.debug(f"FANLayer forward called with src.shape={src.shape}")
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output

@register_model("FANLayerGated")
class FANLayerGated(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, gated=True):
        super(FANLayerGated, self).__init__()
        logger.debug("Initializing FANLayerGated ...")
        self.input_linear_p = nn.Linear(input_dim, output_dim // 4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim - output_dim // 2))
        self.activation = nn.GELU()
        if gated:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
            logger.debug("Gate parameter created for FANLayerGated")

    def forward(self, src):
        logger.debug(f"FANLayerGated forward called with src.shape={src.shape}")
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        if not hasattr(self, "gate"):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate_val = torch.sigmoid(self.gate)
            logger.debug(f"Current gate value (sigmoid): {gate_val.item():.4f}")
            output = torch.cat(
                (gate_val * torch.cos(p), gate_val * torch.sin(p), (1 - gate_val) * g),
                dim=-1
            )
        return output

@register_model("FAN")
class FAN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3):
        super(FAN, self).__init__()
        logger.info("Initializing FAN model ...")
        # For compatibility with time series data
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayer(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        logger.debug(f"FAN forward called with src.shape={src.shape}")
        
        # Explicitly handle time series input with our custom input dimension
        src = _ensure_compatible_input(src, self.input_dim, logger)
        
        # Now we know src has shape [batch_size, input_dim]
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model("FANGated")
class FANGated(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, gated=True):
        super(FANGated, self).__init__()
        logger.info("Initializing FANGated model ...")
        # For compatibility with time series data
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayerGated(hidden_dim, hidden_dim, gated=gated))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        logger.debug(f"FANGated forward called with src.shape={src.shape}")
        
        # Explicitly handle time series input with our custom input dimension
        src = _ensure_compatible_input(src, self.input_dim, logger)
        
        # Now we know src has shape [batch_size, input_dim]
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model("MLP")
class MLPModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, use_embedding=True):
        super(MLPModel, self).__init__()
        logger.info("Initializing MLPModel ...")
        # For compatibility with time series data
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_embedding = use_embedding
        
        self.activation = nn.GELU()
        self.layers = nn.ModuleList()
        if use_embedding:
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        else:
            self.layers.extend([nn.Linear(input_dim, hidden_dim), self.activation])
        for _ in range(num_layers - 2):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        logger.debug(f"MLPModel forward called with src.shape={src.shape}")
        
        # Explicitly handle time series input with our custom input dimension
        src = _ensure_compatible_input(src, self.input_dim, logger)
        
        # Now we know src has shape [batch_size, input_dim]
        output = self.embedding(src) if hasattr(self, "embedding") else src
        for layer in self.layers:
            output = layer(output)
        return output


@register_model("FANLayerOffsetOnlyCos")
class FANLayerOffsetOnlyCos(nn.Module):
    """
    FAN-like layer where we add a per-feature offset only to the cosine part,
    while sine remains unshifted. That is:
        p_cos = p + offset
        out = [cos(p_cos), sin(p), g]
    where p and g are linear transforms of the input.
    """

    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerOffsetOnlyCos, self).__init__()

        # 1) Linear transforms:
        #    output_dim//4 for p (cos, sin) => total 2*(output_dim//4) = output_dim//2
        #    the remaining output_dim//2 for g
        self.input_linear_p = nn.Linear(input_dim, output_dim // 4, bias=bias)
        self.input_linear_g = nn.Linear(
            input_dim, output_dim - (output_dim // 2), bias=bias
        )

        # 2) Activation for g
        self.activation = nn.GELU()

        # 3) Per-feature offset (shape: output_dim//4)
        self.offset = nn.Parameter(torch.zeros(output_dim // 4))

    def forward(self, src):
        """
        src: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        # a) Compute p and g
        p = self.input_linear_p(src)  # (batch_size, output_dim//4)
        g = self.activation(self.input_linear_g(src))  # (batch_size, output_dim//2)

        # b) Only add offset to cosine path
        p_cos = p + self.offset  # broadcast offset across batch

        cos_part = torch.cos(p_cos)
        sin_part = torch.sin(p)

        # c) Concatenate them
        output = torch.cat((cos_part, sin_part, g), dim=-1)
        return output



@register_model("FANOffsetOnlyCosModel")
class FANOffsetOnlyCosModel(nn.Module):
    """
    A top-level model similar to FANGated, but using FANLayerOffsetOnlyCos layers.
    We embed the input to hidden_dim, apply (num_layers - 1) offset-only-cos layers,
    and a final linear to get output_dim.
    """
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True):
        super(FANOffsetOnlyCosModel, self).__init__()

        # 1) Embedding from input_dim -> hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        # 2) Build (num_layers - 1) offset-only-cos layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayerOffsetOnlyCos(hidden_dim, hidden_dim, bias=bias))

        # 3) Final linear: hidden_dim -> output_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        """
        Forward pass:
        src: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        # a) Embed the input
        x = self.embedding(src)

        # b) Pass through each layer in sequence
        for layer in self.layers:
            x = layer(x)

        return x


@register_model("FANLayerPhaseOffset")
class FANLayerPhaseOffset(nn.Module):
    """
    FAN-like layer where each feature dimension i has a trainable phase offset φ_i.
    We output:
        sin(p + φ), cos(p + π/2 - φ), and g
    where p and g are linear transforms of the input.

    p has shape (batch_size, output_dim//4), from which we produce sin(...) and cos(...).
    g has shape (batch_size, output_dim//2), with an activation.
    The final output is concatenated: [sin(p+offset), cos(p+π/2-offset), g].
    """

    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerPhaseOffset, self).__init__()

        # 1) Linear transforms:
        #    output_dim//4 for p => we get sin(...) + cos(...) => total output_dim//2
        #    the remaining output_dim//2 for g
        if output_dim % 2 != 0:
            raise ValueError(
                "Output dimension must be even, so half can be allocated to g."
            )
        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        # 2) Activation for g
        self.activation = nn.GELU()

        # 3) Per-feature offset (shape: p_dim), one offset per feature dimension
        #    so we have φ in shape (p_dim,).
        self.offset = nn.Parameter(torch.full((p_dim,), math.pi / 4))

    def forward(self, src):
        """
        src: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        Specifically: [ sin(p+offset), cos(p + π/2 - offset), g ].
        """
        # a) Compute p and g
        #    p => (batch_size, p_dim)
        #    g => (batch_size, g_dim) then pass through GELU
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        # b) Apply offset only in the sine and cosine channels
        #    For each feature dimension, p_i + offset_i
        #    and p_i + (π/2 - offset_i)
        p_plus = p + self.offset  # used for sin
        p_minus = p + (torch.pi / 2) - self.offset  # used for cos

        sin_part = torch.sin(p_plus)  # (batch_size, p_dim)
        cos_part = torch.cos(p_minus)  # (batch_size, p_dim)

        # c) Concatenate => total dimension = p_dim + p_dim + g_dim = 2*p_dim + g_dim = output_dim
        output = torch.cat((sin_part, cos_part, g), dim=-1)
        return output


@register_model("FANPhaseOffsetModel")
class FANPhaseOffsetModel(nn.Module):
    """
    A top-level model similar to FANGated, but using FANLayerPhaseOffset layers.
    We embed the input to hidden_dim, apply (num_layers - 1) phase-offset layers,
    and a final linear to get output_dim.

    Structure:
      1) Embedding:        (input_dim -> hidden_dim)
      2) (num_layers - 1) x FANLayerPhaseOffset: (hidden_dim -> hidden_dim)
      3) Final linear:     (hidden_dim -> output_dim)
    """

    def __init__(
        self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True
    ):
        super(FANPhaseOffsetModel, self).__init__()

        logger.info("Initializing FANPhaseOffsetModel ...")
        self.input_dim = input_dim  # Save for reference in forward method

        # 1) Embedding from input_dim -> hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        # 2) Build (num_layers - 1) phase-offset layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayerPhaseOffset(hidden_dim, hidden_dim, bias=bias))

        # 3) Final linear: hidden_dim -> output_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        """
        Forward pass:
        src: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        # Simplified approach: expect 2D inputs [batch_size, features]
        # Our data loaders always ensure this format
        
        # If somehow we still get a different shape, handle it
        if src.dim() == 1:
            # 1D input [batch_size] - add feature dimension
            src = src.unsqueeze(1)
        elif src.dim() > 2:
            # Higher dim input - flatten to 2D
            batch_size = src.shape[0]
            src = src.reshape(batch_size, -1)
            # Take only the first feature if too many
            if src.shape[1] > 1:
                src = src[:, :1]
        
        # a) Embed the input
        x = self.embedding(src)

        # b) Pass through each layer in sequence
        for layer in self.layers:
            x = layer(x)

        return x



@register_model("FANLayerPhaseOffsetLimited")
class FANLayerPhaseOffsetLimited(nn.Module):
    """
    A FAN-like layer where each feature dimension has a trainable per-feature
    offset (φ_i). The output is:

        [ sin(p + φ),  cos(p + π/2 - φ),  g ]

    where:
      - p (phase) is a linear transform of shape (batch_size, p_dim).
      - φ is per-feature. If limit_phase_offset=True, φ is restricted to
        [-max_phase, +max_phase] via a tanh re-parameterization.
      - g is another linear transform with GELU activation, of shape (batch_size, output_dim//2).

    The final output shape is (batch_size, output_dim).

    Args:
        input_dim (int):  Dimension of the input.
        output_dim (int): Dimension of the output. Must be divisible by 2.
            Typically, output_dim//4 is used for sin(...) and cos(...),
            and output_dim//2 for g.
        bias (bool): Whether to include bias terms in the linear layers.
        limit_phase_offset (bool): If True, the offset is constrained to [-max_phase, +max_phase].
        max_phase (float): The maximum absolute phase shift if limit_phase_offset is True.
                           Defaults to π.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        limit_phase_offset: bool = False,
        max_phase: float = math.pi,
    ):
        super(FANLayerPhaseOffsetLimited, self).__init__()

        if output_dim % 2 != 0:
            raise ValueError("FANLayerPhaseOffset: output_dim must be even.")

        # We'll split the output into:
        #   p_dim = output_dim // 4 (for sin & cos)
        #   g_dim = output_dim // 2 (for g)
        # Summation: (p_dim + p_dim) + g_dim = output_dim
        p_dim = output_dim // 4
        g_dim = output_dim // 2

        # 1) Linear transforms for p and g
        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)
        self.activation = nn.GELU()

        # 2) Phase Offset Parameters
        #    We'll store an unconstrained offset parameter, and
        #    optionally transform it with tanh in the forward pass.
        self.limit_phase_offset = limit_phase_offset
        self.max_phase = max_phase
        self.offset_unconstrained = nn.Parameter(torch.zeros(p_dim))

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:

        Args:
            src (Tensor): shape (batch_size, input_dim)

        Returns:
            Tensor of shape (batch_size, output_dim):
                [ sin(p + offset), cos(p + π/2 - offset), g ].
        """
        # a) Compute p and g
        p = self.input_linear_p(src)  # (batch_size, p_dim)
        g_unactivated = self.input_linear_g(src)  # (batch_size, g_dim)
        g = self.activation(g_unactivated)

        # b) Determine the actual offset
        #    If limit_phase_offset=True, constrain to [-max_phase, +max_phase].
        if self.limit_phase_offset:
            offset = self.max_phase * torch.tanh(self.offset_unconstrained)
        else:
            offset = self.offset_unconstrained

        # c) Compute phases for sine & cosine
        p_plus = p + offset  # used for sin
        p_minus = p + (math.pi / 2) - offset  # used for cos

        sin_part = torch.sin(p_plus)  # (batch_size, p_dim)
        cos_part = torch.cos(p_minus)  # (batch_size, p_dim)

        # d) Concatenate => shape = (batch_size, output_dim)
        output = torch.cat([sin_part, cos_part, g], dim=-1)
        return output



@register_model("FANPhaseOffsetModelZero")
class FANPhaseOffsetModelZeros(nn.Module):
    """
    A top-level model similar to FANGated, but using FANLayerPhaseOffset.

    Structure:
      1) Embedding (input_dim -> hidden_dim)
      2) (num_layers - 1) x FANLayerPhaseOffset (hidden_dim -> hidden_dim)
      3) Final Linear (hidden_dim -> output_dim)

    Args:
        input_dim (int):  Input dimension.
        output_dim (int): Output dimension for the final layer.
        hidden_dim (int): Dimension of hidden layers.
        num_layers (int): Number of total layers, must be >= 2.
        bias (bool):      Include bias in linear layers.

        limit_phase_offset (bool): Whether to constrain offsets to [-max_phase, +max_phase].
        max_phase (float): The bounding range for offsets if limit_phase_offset=True.
                           Defaults to π.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        bias: bool = True,
        limit_phase_offset: bool = False,
        max_phase: float = math.pi,
    ):
        super(FANPhaseOffsetModelZeros, self).__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 (embedding + final at minimum).")

        logger.info("Initializing FANPhaseOffsetModelZeros ...")

        # 1) Embedding from input_dim -> hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        # 2) Build (num_layers - 1) FANLayerPhaseOffset layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerPhaseOffsetZero(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    bias=bias,
                    limit_phase_offset=limit_phase_offset,
                    max_phase=max_phase,
                )
            )

        # 3) Final Linear (hidden_dim -> output_dim)
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:

        Args:
            src (Tensor): shape (batch_size, input_dim) or (batch_size,) when input_dim=1
                        or potentially (batch_size, 1, 1) from some data pipelines

        Returns:
            Tensor: shape (batch_size, output_dim)
        """
        # Ensure input has proper dimensions
        if src.dim() == 1:
            # If input is 1D (batch_size,), unsqueeze to (batch_size, 1)
            src = src.unsqueeze(-1)
        elif src.dim() == 3:
            # If input is 3D (batch_size, 1, 1), reshape to (batch_size, 1)
            src = src.squeeze(1)
        elif src.dim() > 3:
            # Log warning if unexpected input shape
            logger.warning(f"Unexpected input shape: {src.shape}, expected 2D tensor")
        
        # a) Embed the input
        x = self.embedding(src)

        # b) Pass through the phase-offset layers, then the final linear
        for layer in self.layers:
            x = layer(x)

        return x


@register_model("FANLayerPhaseOffsetGated")
class FANLayerPhaseOffsetGated(nn.Module):
    """
    Gated FAN-like layer with per-feature phase offset.

    For an input \(\mathbf{x}\), we compute:
      - \( \mathbf{p} = W_p \mathbf{x} + b_p \), with \(\mathbf{p} \in \mathbb{R}^{p_{\text{dim}}} \) (where \( p_{\text{dim}} = \text{output_dim} / 4 \)).
      - \( \mathbf{g} = \text{GELU}(W_g \mathbf{x} + b_g) \), with \(\mathbf{g} \in \mathbb{R}^{g_{\text{dim}}} \) (where \( g_{\text{dim}} = \text{output_dim} / 2 \)).

    Then, using a learned per-feature phase offset \( \phi \) and a scalar gate \( s \) (with \( \gamma = \sigma(s) \)):

      - \( \mathbf{p}_+ = \mathbf{p} + \phi \) is used in the sine branch.
      - \( \mathbf{p}_- = \mathbf{p} + \frac{\pi}{2} - \phi \) is used in the cosine branch.

    The final output is:
      \[
      \mathbf{y} = \text{Concat}\Bigl( \gamma \cdot \sin(\mathbf{p}_+),\; \gamma \cdot \cos(\mathbf{p}_-),\; (1-\gamma) \cdot \mathbf{g} \Bigr)
      \]
    ensuring that the Fourier branch (with phase adjustments) and the nonlinear branch are dynamically balanced.
    """

    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerPhaseOffsetGated, self).__init__()

        if output_dim % 2 != 0:
            raise ValueError("Output dimension must be even.")

        # Define dimensions:
        p_dim = output_dim // 4  # For phase (sin/cos) branch
        g_dim = output_dim // 2  # For nonlinear branch

        # Linear projections for p and g
        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)
        self.activation = nn.GELU()

        # Per-feature phase offset: φ ∈ ℝ^(p_dim)
        self.offset = nn.Parameter(torch.full((p_dim,), math.pi / 4))

        # Learnable scalar gate parameter: s ∈ ℝ, gate = σ(s)
        self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, src):
        # Compute projections
        p = self.input_linear_p(src)  # Shape: (batch_size, p_dim)
        g = self.activation(self.input_linear_g(src))  # Shape: (batch_size, g_dim)

        # Compute adjusted phases:
        p_plus = p + self.offset  # For sine
        p_minus = p + (torch.pi / 2) - self.offset  # For cosine

        # Trigonometric branch:
        sin_part = torch.sin(p_plus)
        cos_part = torch.cos(p_minus)

        # Compute gate value γ = sigmoid(s)
        gamma = torch.sigmoid(self.gate)

        # Apply gating:
        fourier_branch = gamma * torch.cat((sin_part, cos_part), dim=-1)
        nonlinear_branch = (1 - gamma) * g

        # Concatenate branches to form final output:
        output = torch.cat((fourier_branch, nonlinear_branch), dim=-1)
        return output



@register_model("FANLayerPhaseOffsetZero")
class FANLayerPhaseOffsetZero(nn.Module):
    """
    FAN-like layer where each feature dimension i has a trainable phase offset φ_i.
    We output:
        sin(p + φ), cos(p + π/2 - φ), and g
    where p and g are linear transforms of the input.

    p has shape (batch_size, output_dim//4), from which we produce sin(...) and cos(...).
    g has shape (batch_size, output_dim//2), with an activation.
    The final output is concatenated: [sin(p+offset), cos(p+π/2-offset), g].
    """

    def __init__(self, input_dim, output_dim, bias=True, limit_phase_offset=False, max_phase=math.pi):
        super(FANLayerPhaseOffsetZero, self).__init__()

        # 1) Linear transforms:
        #    output_dim//4 for p => we get sin(...) + cos(...) => total output_dim//2
        #    the remaining output_dim//2 for g
        if output_dim % 2 != 0:
            raise ValueError(
                "Output dimension must be even, so half can be allocated to g."
            )
        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        # 2) Activation for g
        self.activation = nn.GELU()

        # Save parameters
        self.limit_phase_offset = limit_phase_offset
        self.max_phase = max_phase
        
        # make zeros
        self.offset = nn.Parameter(torch.zeros(p_dim))

    def forward(self, src):
        """
        src: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        Specifically: [ sin(p+offset), cos(p + π/2 - offset), g ].
        """
        # a) Compute p and g
        #    p => (batch_size, p_dim)
        #    g => (batch_size, g_dim) then pass through GELU
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        # b) Apply offset only in the sine and cosine channels
        #    For each feature dimension, p_i + offset_i
        #    and p_i + (π/2 - offset_i)
        p_plus = p + self.offset  # used for sin
        p_minus = p + (torch.pi / 2) - self.offset  # used for cos

        sin_part = torch.sin(p_plus)  # (batch_size, p_dim)
        cos_part = torch.cos(p_minus)  # (batch_size, p_dim)

        # c) Concatenate => total dimension = p_dim + p_dim + g_dim = 2*p_dim + g_dim = output_dim
        output = torch.cat((sin_part, cos_part, g), dim=-1)
        return output


@register_model("FANPhaseOffsetModelGated")
class FANPhaseOffsetModelGated(nn.Module):
    """
    A top-level model that uses gated FANLayerPhaseOffsetGated layers.

    Structure:
      1) Embedding:        (input_dim -> hidden_dim)
      2) (num_layers - 1) x FANLayerPhaseOffsetGated: (hidden_dim -> hidden_dim)
      3) Final linear:     (hidden_dim -> output_dim)
    """

    def __init__(
        self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True
    ):
        super(FANPhaseOffsetModelGated, self).__init__()
        logger.info("Initializing FANPhaseOffsetModelGated ...")

        # 1) Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        # 2) Stacked gated phase-offset layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerPhaseOffsetGated(hidden_dim, hidden_dim, bias=bias)
            )

        # 3) Final linear layer mapping hidden_dim -> output_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        # a) Embed input
        x = self.embedding(src)
        # b) Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x)
        return x


@register_model("FANLayerAmplitudePhase")
class FANLayerAmplitudePhase(nn.Module):
    """
    FAN-like layer with trainable phase offset φ_i and amplitude A_i for each feature dimension i.
    We output:
        A*sin(p + φ), A*cos(p + π/2 - φ), and g
    where p and g are linear transforms of the input.

    p has shape (batch_size, output_dim//4), from which we produce sin(...) and cos(...).
    g has shape (batch_size, output_dim//2), with an activation.
    The final output is concatenated: [A*sin(p+offset), A*cos(p+π/2-offset), g].
    """

    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerAmplitudePhase, self).__init__()

        # 1) Linear transforms:
        #    output_dim//4 for p => we get sin(...) + cos(...) => total output_dim//2
        #    the remaining output_dim//2 for g
        if output_dim % 2 != 0:
            raise ValueError(
                "Output dimension must be even, so half can be allocated to g."
            )
        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        # 2) Activation for g
        self.activation = nn.GELU()

        # 3) Per-feature phase offset (shape: p_dim), one offset per feature dimension
        self.offset = nn.Parameter(torch.full((p_dim,), math.pi / 4))

        # 4) Per-feature amplitude (shape: p_dim), one amplitude per feature dimension
        #    Initialize to 1.0 for each dimension
        self.amplitude = nn.Parameter(torch.ones(p_dim))

    def forward(self, src):
        """
        src: (batch_size, input_dim) or potentially (batch_size, 1, input_dim) or (batch_size, seq_len, input_dim)
        returns: (batch_size, output_dim)
        Specifically: [ A*sin(p+offset), A*cos(p + π/2 - offset), g ].
        """
        # Ensure input has compatible shape using common helper function
        src = _ensure_compatible_input(src, self.input_linear_p.in_features, logger)
                
        # a) Compute p and g
        #    p => (batch_size, p_dim)
        #    g => (batch_size, g_dim) then pass through GELU
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        # b) Apply offset and amplitude in the sine and cosine channels
        p_plus = p + self.offset  # used for sin
        p_minus = p + (torch.pi / 2) - self.offset  # used for cos

        sin_part = self.amplitude * torch.sin(p_plus)  # (batch_size, p_dim)
        cos_part = self.amplitude * torch.cos(p_minus)  # (batch_size, p_dim)

        # c) Concatenate => total dimension = p_dim + p_dim + g_dim = 2*p_dim + g_dim = output_dim
        output = torch.cat((sin_part, cos_part, g), dim=-1)
        return output


@register_model("FANAmplitudePhaseModel")
class FANAmplitudePhaseModel(nn.Module):
    """
    A top-level model using FANLayerAmplitudePhase layers.
    We embed the input to hidden_dim, apply (num_layers - 1) amplitude-phase layers,
    and a final linear to get output_dim.

    Structure:
      1) Embedding:        (input_dim -> hidden_dim)
      2) (num_layers - 1) x FANLayerAmplitudePhase: (hidden_dim -> hidden_dim)
      3) Final linear:     (hidden_dim -> output_dim)
    """

    def __init__(
        self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True
    ):
        super(FANAmplitudePhaseModel, self).__init__()

        logger.info("Initializing FANAmplitudePhaseModel ...")
        # Store dimensions for shape validation
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 1) Embedding from input_dim -> hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        # 2) Build (num_layers - 1) amplitude-phase layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerAmplitudePhase(hidden_dim, hidden_dim, bias=bias)
            )

        # 3) Final linear: hidden_dim -> output_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        """
        Forward pass:
        src: (batch_size, input_dim) or potentially (batch_size, 1, input_dim) or (batch_size, seq_len, input_dim)
        returns: (batch_size, output_dim)
        """
        # Ensure input has compatible shape using common helper function
        src = _ensure_compatible_input(src, self.embedding.in_features, logger)
        
        # a) Embed the input
        x = self.embedding(src)

        # b) Pass through each layer in sequence
        for layer in self.layers:
            x = layer(x)

        # Reshape targets if necessary to match expected output dimension (important fix)
        if hasattr(self, 'output_dim') and x.shape[-1] != self.output_dim:
            logger.warning(f"Reshaping output from {x.shape} to match output_dim={self.output_dim}")
            if x.shape[-1] > self.output_dim:
                # Trim excess dimensions
                x = x[..., :self.output_dim]
            elif x.shape[-1] < self.output_dim:
                # Pad with zeros
                padding = torch.zeros(*x.shape[:-1], self.output_dim - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)

        return x



@register_model("FANLayerMultiFrequency")
class FANLayerMultiFrequency(nn.Module):
    """
    FAN-like layer supporting multiple frequency components.
    For each input dimension, we compute sinusoidal outputs at different frequency scales.

    We output:
        For each frequency i: sin(f_i*p + φ_i), cos(f_i*p + π/2 - φ_i)
        Plus the gating component g
    where p and g are linear transforms of the input, and f_i are learnable frequency scales.
    """

    def __init__(self, input_dim, output_dim, n_freqs=2, bias=True):
        super(FANLayerMultiFrequency, self).__init__()

        # Check that dimensions are compatible with frequency count
        if output_dim % 2 != 0:
            raise ValueError("Output dimension must be even.")

        # Ensure we can divide the output dimensions properly
        trig_part = output_dim // 2  # Half for trig functions, half for g
        if trig_part % n_freqs != 0:
            raise ValueError(
                f"Trigonometric part (output_dim//2 = {trig_part}) must be divisible by n_freqs ({n_freqs})"
            )

        # Calculate dimensions
        p_dim = trig_part // n_freqs  # For each frequency we need sin and cos
        g_dim = output_dim - trig_part

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        # Activation for g
        self.activation = nn.GELU()

        # Per-feature phase offset for each frequency
        # Shape: (n_freqs, p_dim)
        self.offset = nn.Parameter(torch.full((n_freqs, p_dim), math.pi / 4))

        # Learnable frequency scales - initialize with log spacing
        # Shape: (n_freqs,)
        self.freq_scales = nn.Parameter(torch.logspace(-0.5, 0.5, n_freqs))

        # Store dimensions for forward pass
        self.n_freqs = n_freqs
        self.p_dim = p_dim
        self.g_dim = g_dim

    def forward(self, src):
        """
        src: (batch_size, input_dim) or potentially (batch_size, 1, input_dim) or (batch_size, seq_len, input_dim)
        returns: (batch_size, output_dim)
        Output combines multiple frequency sinusoidal components and the gating part.
        """
        # Ensure input has compatible shape using common helper function
        src = _ensure_compatible_input(src, self.input_linear_p.in_features, logger)
                
        # Compute base p and g
        p = self.input_linear_p(src)  # (batch_size, p_dim)
        g = self.activation(self.input_linear_g(src))  # (batch_size, g_dim)

        # Prepare list for concatenation
        trig_parts = []

        # For each frequency scale
        for i in range(self.n_freqs):
            # Scale p by the frequency factor
            scaled_p = p * self.freq_scales[i]

            # Apply phase offset
            p_plus = scaled_p + self.offset[i]  # for sin
            p_minus = scaled_p + (torch.pi / 2) - self.offset[i]  # for cos

            # Compute trigonometric functions
            sin_part = torch.sin(p_plus)  # (batch_size, p_dim)
            cos_part = torch.cos(p_minus)  # (batch_size, p_dim)

            # Add to list
            trig_parts.append(sin_part)
            trig_parts.append(cos_part)

        # Concatenate all parts
        output = torch.cat(trig_parts + [g], dim=-1)
        return output


@register_model("FANMultiFrequencyModel")
class FANMultiFrequencyModel(nn.Module):
    """
    A top-level model using FANLayerMultiFrequency layers.
    We embed the input to hidden_dim, apply (num_layers - 1) multi-frequency layers,
    and a final linear to get output_dim.

    Structure:
      1) Embedding:        (input_dim -> hidden_dim)
      2) (num_layers - 1) x FANLayerMultiFrequency: (hidden_dim -> hidden_dim)
      3) Final linear:     (hidden_dim -> output_dim)
    """

    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        hidden_dim=2048,
        num_layers=3,
        n_freqs=2,
        bias=True,
    ):
        super(FANMultiFrequencyModel, self).__init__()

        logger.info(
            "Initializing FANMultiFrequencyModel with %d frequencies...", n_freqs
        )

        # 1) Embedding from input_dim -> hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        # 2) Build (num_layers - 1) multi-frequency layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerMultiFrequency(
                    hidden_dim, hidden_dim, n_freqs=n_freqs, bias=bias
                )
            )

        # 3) Final linear: hidden_dim -> output_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        """
        Forward pass:
        src: (batch_size, input_dim) or potentially (batch_size, 1, input_dim) or (batch_size, seq_len, input_dim)
        returns: (batch_size, output_dim)
        """
        # Ensure input has compatible shape
        src = _ensure_compatible_input(src, self.embedding.in_features, logger)
        
        # a) Embed the input
        x = self.embedding(src)

        # b) Pass through each layer in sequence
        for layer in self.layers:
            x = layer(x)

        return x


@register_model("FANLayerUnconstrainedBasis")
class FANLayerUnconstrainedBasis(nn.Module):
    """
    FAN-like layer using a fully parameterized 2x2 transformation matrix
    for each feature dimension, allowing unconstrained linear transformations
    of the sin/cos basis.

    We output:
        T[sin(p), cos(p)] and g
    where p and g are linear transforms of the input, and T is a learned 2x2 matrix
    per feature dimension.
    """

    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerUnconstrainedBasis, self).__init__()

        # 1) Linear transforms
        if output_dim % 2 != 0:
            raise ValueError(
                "Output dimension must be even, so half can be allocated to g."
            )
        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        # 2) Activation for g
        self.activation = nn.GELU()

        # 3) Per-feature basis transformation (shape: p_dim, 2, 2)
        # Initialize as identity-like matrices with small perturbation
        basis = torch.stack([torch.eye(2) for _ in range(p_dim)])
        basis = basis + torch.randn_like(basis) * 0.01
        self.basis_transform = nn.Parameter(basis)

    def forward(self, src):
        """
        src: (batch_size, input_dim) or potentially (batch_size, 1, input_dim)
        returns: (batch_size, output_dim)
        """
        # Ensure input has compatible shape using common helper function
        src = _ensure_compatible_input(src, self.input_linear_p.in_features, logger)
            
        # a) Compute p and g
        p = self.input_linear_p(src)  # (batch_size, p_dim)
        g = self.activation(self.input_linear_g(src))  # (batch_size, g_dim)

        # b) Compute sin and cos of p
        sin_p = torch.sin(p)  # (batch_size, p_dim)
        cos_p = torch.cos(p)  # (batch_size, p_dim)

        # Stack them for matrix multiplication
        # Shape: (batch_size, p_dim, 2)
        trig_inputs = torch.stack([sin_p, cos_p], dim=-1)
        
        # Debug log the shapes
        logger.debug(f"trig_inputs shape: {trig_inputs.shape}, basis_transform shape: {self.basis_transform.shape}")

        # Apply basis transformation
        # Shape of basis_transform: (p_dim, 2, 2)
        # (batch_size, p_dim, 2) @ (p_dim, 2, 2) -> (batch_size, p_dim, 2)
        transformed = torch.einsum("bpd,pdo->bpo", trig_inputs, self.basis_transform)

        # Extract the two components
        # Both shape: (batch_size, p_dim)
        sin_transformed = transformed[:, :, 0]
        cos_transformed = transformed[:, :, 1]

        # c) Concatenate all parts
        output = torch.cat((sin_transformed, cos_transformed, g), dim=-1)
        return output


@register_model("FANUnconstrainedBasisModel")
class FANUnconstrainedBasisModel(nn.Module):
    """
    A top-level model using FANLayerUnconstrainedBasis layers.
    We embed the input to hidden_dim, apply (num_layers - 1) unconstrained basis layers,
    and a final linear to get output_dim.

    Structure:
      1) Embedding:        (input_dim -> hidden_dim)
      2) (num_layers - 1) x FANLayerUnconstrainedBasis: (hidden_dim -> hidden_dim)
      3) Final linear:     (hidden_dim -> output_dim)
    """

    def __init__(
        self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True
    ):
        super(FANUnconstrainedBasisModel, self).__init__()

        logger.info("Initializing FANUnconstrainedBasisModel ...")

        # 1) Embedding from input_dim -> hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        # 2) Build (num_layers - 1) unconstrained basis layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerUnconstrainedBasis(hidden_dim, hidden_dim, bias=bias)
            )

        # 3) Final linear: hidden_dim -> output_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        """
        Forward pass:
        src: (batch_size, input_dim) or potentially (batch_size, 1, input_dim) or (batch_size, sequence_len, input_dim)
        returns: (batch_size, output_dim)
        """
        # Ensure input has compatible shape using common helper function
        src = _ensure_compatible_input(src, self.embedding.in_features, logger)
        
        # a) Embed the input
        x = self.embedding(src)

        # b) Pass through each layer in sequence
        for layer in self.layers:
            x = layer(x)

        return x