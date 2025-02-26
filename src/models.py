# src/src.py
import torch
from torch import nn
from loguru import logger

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