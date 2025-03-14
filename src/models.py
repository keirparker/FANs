from loguru import logger
import torch
import torch.nn as nn
import math

model_registry = {}

def register_model(model_name):
    def decorator(cls):
        model_registry[model_name] = cls
        logger.debug(f"Registered model class: {model_name}")
        return cls
    return decorator

def get_model_by_name(model_name, *args, **kwargs):
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

    model_cls = model_registry.get(model_name)
    if model_cls is None:
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
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model("FANGated")
class FANGated(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, gated=True):
        super(FANGated, self).__init__()
        logger.info("Initializing FANGated model ...")
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
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model("MLP")
class MLPModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, use_embedding=True):
        super(MLPModel, self).__init__()
        logger.info("Initializing MLPModel ...")
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
        output = self.embedding(src) if hasattr(self, "embedding") else src
        for layer in self.layers:
            output = layer(output)
        return output


@register_model("FANLayerOffsetOnlyCos")
class FANLayerOffsetOnlyCos(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerOffsetOnlyCos, self).__init__()

        self.input_linear_p = nn.Linear(input_dim, output_dim // 4, bias=bias)
        self.input_linear_g = nn.Linear(
            input_dim, output_dim - (output_dim // 2), bias=bias
        )

        self.activation = nn.GELU()
        self.offset = nn.Parameter(torch.zeros(output_dim // 4))

    def forward(self, src):
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        p_cos = p + self.offset

        cos_part = torch.cos(p_cos)
        sin_part = torch.sin(p)

        output = torch.cat((cos_part, sin_part, g), dim=-1)
        return output


@register_model("FANOffsetOnlyCosModel")
class FANOffsetOnlyCosModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True):
        super(FANOffsetOnlyCosModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayerOffsetOnlyCos(hidden_dim, hidden_dim, bias=bias))

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x)

        return x


@register_model("FANLayerPhaseOffset")
class FANLayerPhaseOffset(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerPhaseOffset, self).__init__()

        if output_dim % 2 != 0:
            raise ValueError(
                "Output dimension must be even, so half can be allocated to g."
            )
        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        self.activation = nn.GELU()
        self.offset = nn.Parameter(torch.full((p_dim,), math.pi / 4))

    def forward(self, src):
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        p_plus = p + self.offset
        p_minus = p + (torch.pi / 2) - self.offset

        sin_part = torch.sin(p_plus)
        cos_part = torch.cos(p_minus)

        output = torch.cat((sin_part, cos_part, g), dim=-1)
        return output


@register_model("FANPhaseOffsetModel")
class FANPhaseOffsetModel(nn.Module):
    def __init__(
            self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True
    ):
        super(FANPhaseOffsetModel, self).__init__()

        logger.info("Initializing FANPhaseOffsetModel ...")
        self.input_dim = input_dim

        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayerPhaseOffset(hidden_dim, hidden_dim, bias=bias))

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        if src.dim() == 1:
            src = src.unsqueeze(1)
        elif src.dim() > 2:
            batch_size = src.shape[0]
            src = src.reshape(batch_size, -1)
            if src.shape[1] > 1:
                src = src[:, :1]

        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x)

        return x


@register_model("FANLayerPhaseOffsetLimited")
class FANLayerPhaseOffsetLimited(nn.Module):
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

        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)
        self.activation = nn.GELU()

        self.limit_phase_offset = limit_phase_offset
        self.max_phase = max_phase
        self.offset_unconstrained = nn.Parameter(torch.zeros(p_dim))

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        p = self.input_linear_p(src)
        g_unactivated = self.input_linear_g(src)
        g = self.activation(g_unactivated)

        if self.limit_phase_offset:
            offset = self.max_phase * torch.tanh(self.offset_unconstrained)
        else:
            offset = self.offset_unconstrained

        p_plus = p + offset
        p_minus = p + (math.pi / 2) - offset

        sin_part = torch.sin(p_plus)
        cos_part = torch.cos(p_minus)

        output = torch.cat([sin_part, cos_part, g], dim=-1)
        return output


@register_model("FANPhaseOffsetModelZero")
class FANPhaseOffsetModelZeros(nn.Module):
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

        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

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

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if src.dim() == 1:
            src = src.unsqueeze(-1)
        elif src.dim() == 3:
            src = src.squeeze(1)
        elif src.dim() > 3:
            logger.warning(f"Unexpected input shape: {src.shape}, expected 2D tensor")

        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x)

        return x


@register_model("FANLayerPhaseOffsetGated")
class FANLayerPhaseOffsetGated(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerPhaseOffsetGated, self).__init__()

        if output_dim % 2 != 0:
            raise ValueError("Output dimension must be even.")

        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)
        self.activation = nn.GELU()

        self.offset = nn.Parameter(torch.full((p_dim,), math.pi / 4))
        self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, src):
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        p_plus = p + self.offset
        p_minus = p + (torch.pi / 2) - self.offset

        sin_part = torch.sin(p_plus)
        cos_part = torch.cos(p_minus)

        gamma = torch.sigmoid(self.gate)

        fourier_branch = gamma * torch.cat((sin_part, cos_part), dim=-1)
        nonlinear_branch = (1 - gamma) * g

        output = torch.cat((fourier_branch, nonlinear_branch), dim=-1)
        return output


@register_model("FANPhaseOffsetModelGated")
class FANPhaseOffsetModelGated(nn.Module):
    def __init__(
            self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True
    ):
        super(FANPhaseOffsetModelGated, self).__init__()
        logger.info("Initializing FANPhaseOffsetModelGated ...")

        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerPhaseOffsetGated(hidden_dim, hidden_dim, bias=bias)
            )

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        x = self.embedding(src)
        for layer in self.layers:
            x = layer(x)
        return x


@register_model("FANLayerPhaseOffsetZero")
class FANLayerPhaseOffsetZero(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, limit_phase_offset=False, max_phase=math.pi):
        super(FANLayerPhaseOffsetZero, self).__init__()

        if output_dim % 2 != 0:
            raise ValueError(
                "Output dimension must be even, so half can be allocated to g."
            )
        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        self.activation = nn.GELU()

        self.limit_phase_offset = limit_phase_offset
        self.max_phase = max_phase

        self.offset = nn.Parameter(torch.zeros(p_dim))

    def forward(self, src):
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        p_plus = p + self.offset
        p_minus = p + (torch.pi / 2) - self.offset

        sin_part = torch.sin(p_plus)
        cos_part = torch.cos(p_minus)

        output = torch.cat((sin_part, cos_part, g), dim=-1)
        return output


@register_model("FANLayerAmplitudePhase")
class FANLayerAmplitudePhase(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerAmplitudePhase, self).__init__()

        if output_dim % 2 != 0:
            raise ValueError(
                "Output dimension must be even, so half can be allocated to g."
            )
        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        self.activation = nn.GELU()
        self.offset = nn.Parameter(torch.full((p_dim,), math.pi / 4))
        self.amplitude = nn.Parameter(torch.ones(p_dim))

    def forward(self, src):
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        p_plus = p + self.offset
        p_minus = p + (torch.pi / 2) - self.offset

        sin_part = self.amplitude * torch.sin(p_plus)
        cos_part = self.amplitude * torch.cos(p_minus)

        output = torch.cat((sin_part, cos_part, g), dim=-1)
        return output


@register_model("FANAmplitudePhaseModel")
class FANAmplitudePhaseModel(nn.Module):
    def __init__(
            self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True
    ):
        super(FANAmplitudePhaseModel, self).__init__()

        logger.info("Initializing FANAmplitudePhaseModel ...")
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerAmplitudePhase(hidden_dim, hidden_dim, bias=bias)
            )

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x)

        if hasattr(self, 'output_dim') and x.shape[-1] != self.output_dim:
            logger.warning(f"Reshaping output from {x.shape} to match output_dim={self.output_dim}")
            if x.shape[-1] > self.output_dim:
                x = x[..., :self.output_dim]
            elif x.shape[-1] < self.output_dim:
                padding = torch.zeros(*x.shape[:-1], self.output_dim - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)

        return x


@register_model("FANLayerMultiFrequency")
class FANLayerMultiFrequency(nn.Module):
    def __init__(self, input_dim, output_dim, n_freqs=2, bias=True):
        super(FANLayerMultiFrequency, self).__init__()

        if output_dim % 2 != 0:
            raise ValueError("Output dimension must be even.")

        trig_part = output_dim // 2
        if trig_part % n_freqs != 0:
            raise ValueError(
                f"Trigonometric part (output_dim//2 = {trig_part}) must be divisible by n_freqs ({n_freqs})"
            )

        p_dim = trig_part // n_freqs
        g_dim = output_dim - trig_part

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        self.activation = nn.GELU()
        self.offset = nn.Parameter(torch.full((n_freqs, p_dim), math.pi / 4))
        self.freq_scales = nn.Parameter(torch.logspace(-0.5, 0.5, n_freqs))

        self.n_freqs = n_freqs
        self.p_dim = p_dim
        self.g_dim = g_dim

    def forward(self, src):
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        trig_parts = []

        for i in range(self.n_freqs):
            scaled_p = p * self.freq_scales[i]

            p_plus = scaled_p + self.offset[i]
            p_minus = scaled_p + (torch.pi / 2) - self.offset[i]

            sin_part = torch.sin(p_plus)
            cos_part = torch.cos(p_minus)

            trig_parts.append(sin_part)
            trig_parts.append(cos_part)

        output = torch.cat(trig_parts + [g], dim=-1)
        return output


@register_model("FANMultiFrequencyModel")
class FANMultiFrequencyModel(nn.Module):
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

        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerMultiFrequency(
                    hidden_dim, hidden_dim, n_freqs=n_freqs, bias=bias
                )
            )

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x)

        return x


@register_model("FANLayerUnconstrainedBasis")
class FANLayerUnconstrainedBasis(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayerUnconstrainedBasis, self).__init__()

        if output_dim % 2 != 0:
            raise ValueError(
                "Output dimension must be even, so half can be allocated to g."
            )
        p_dim = output_dim // 4
        g_dim = output_dim // 2

        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim, bias=bias)

        self.activation = nn.GELU()

        basis = torch.stack([torch.eye(2) for _ in range(p_dim)])
        basis = basis + torch.randn_like(basis) * 0.01
        self.basis_transform = nn.Parameter(basis)

    def forward(self, src):
        p = self.input_linear_p(src)
        g = self.activation(self.input_linear_g(src))

        sin_p = torch.sin(p)
        cos_p = torch.cos(p)

        trig_inputs = torch.stack([sin_p, cos_p], dim=-1)

        logger.debug(f"trig_inputs shape: {trig_inputs.shape}, basis_transform shape: {self.basis_transform.shape}")

        transformed = torch.einsum("bpd,pdo->bpo", trig_inputs, self.basis_transform)

        sin_transformed = transformed[:, :, 0]
        cos_transformed = transformed[:, :, 1]

        output = torch.cat((sin_transformed, cos_transformed, g), dim=-1)
        return output


@register_model("FANUnconstrainedBasisModel")
class FANUnconstrainedBasisModel(nn.Module):
    def __init__(
            self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, bias=True
    ):
        super(FANUnconstrainedBasisModel, self).__init__()

        logger.info("Initializing FANUnconstrainedBasisModel ...")

        self.embedding = nn.Linear(input_dim, hidden_dim, bias=bias)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                FANLayerUnconstrainedBasis(hidden_dim, hidden_dim, bias=bias)
            )

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, src):
        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x)

        return x