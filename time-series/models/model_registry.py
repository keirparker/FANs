from typing import Dict, Type, Any, Callable, Optional
import torch.nn as nn
from loguru import logger

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str) -> Callable:
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _MODEL_REGISTRY:
            logger.warning(f"Model {name} is already registered. Overwriting previous registration.")
        _MODEL_REGISTRY[name] = cls
        cls._model_name = name
        logger.debug(f"Registered model: {name}")
        return cls
    return decorator

def get_model_by_name(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        available_models = sorted(list(_MODEL_REGISTRY.keys()))
        error_msg = f"No model found with name '{name}'. Available models: {available_models}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    model_cls = _MODEL_REGISTRY[name]
    logger.info(f"Creating model: {name} with args: {kwargs}")
    return model_cls(**kwargs)

def list_available_models() -> list:
    return sorted(list(_MODEL_REGISTRY.keys()))