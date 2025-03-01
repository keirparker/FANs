# src/src.py
from loguru import logger
import torch
import torch.nn as nn
import math


# Model registry dictionary
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
    model_cls = model_registry.get(model_name)
    if model_cls is None:
        logger.error(f"No model found with model_name {model_name}.")
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
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayer(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        logger.debug(f"FAN forward called with src.shape={src.shape}")
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model("FANGated")
class FANGated(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, gated=True):
        super(FANGated, self).__init__()
        logger.info("Initializing FANGated model ...")
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayerGated(hidden_dim, hidden_dim, gated=gated))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        logger.debug(f"FANGated forward called with src.shape={src.shape}")
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model("MLP")
class MLPModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, use_embedding=True):
        super(MLPModel, self).__init__()
        logger.info("Initializing MLPModel ...")
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


@register_model("FANPhaseOffsetModelLimited")
class FANPhaseOffsetModelLimited(nn.Module):
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
        super(FANPhaseOffsetModelLimited, self).__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 (embedding + final at minimum).")

        logger.info("Initializing FANPhaseOffsetModel ...")

        # 1) Embedding from input_dim -> hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        # 2) Build (num_layers - 1) FANLayerPhaseOffset layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerPhaseOffsetLimited(
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
            src (Tensor): shape (batch_size, input_dim)

        Returns:
            Tensor: shape (batch_size, output_dim)
        """
        # a) Embed the input
        x = self.embedding(src)

        # b) Pass through the phase-offset layers, then the final linear
        for layer in self.layers:
            x = layer(x)

        return x

import torch
import torch.nn as nn
import math
import logging
from src.models import register_model

logger = logging.getLogger(__name__)


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
        self.offset = nn.Parameter(torch.zeros(p_dim))

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
