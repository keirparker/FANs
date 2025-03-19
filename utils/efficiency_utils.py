#!/usr/bin/env python
"""Efficiency utilities for measuring model performance metrics."""

import time
import torch
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple
import torch.nn as nn

# Try to import thop for FLOPs counting
try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    logger.warning("thop package not found. FLOPs counting will be estimated.")
    THOP_AVAILABLE = False


def count_params(model: nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_size: Tuple) -> int:
    """Count the number of FLOPs for a single forward pass."""
    dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
    
    if THOP_AVAILABLE:
        flops, _ = thop_profile(model, inputs=(dummy_input,))
        return int(flops)
    else:
        param_count = count_params(model)
        return param_count * 3


def measure_inference_time(model: nn.Module, input_size: Tuple, num_repeats: int = 100) -> float:
    """Measure the average inference time in milliseconds."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_repeats):
        _ = model(dummy_input)
        
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    return (end_time - start_time) * 1000 / num_repeats


def compute_efficiency_metrics(
    model: nn.Module, 
    input_size: Tuple,
    training_time_seconds: float,
    num_parameters: Optional[int] = None,
    num_epochs: Optional[int] = None,
    dataset_size: Optional[int] = None
) -> Dict[str, float]:
    """Compute comprehensive efficiency metrics for a model."""
    if num_parameters is None:
        num_parameters = count_params(model)
    
    inference_time_ms = measure_inference_time(model, input_size)
    flops = count_flops(model, input_size)
    mflops = flops / 1e6
    
    metrics = {
        "num_parameters": num_parameters,
        "inference_time_ms": inference_time_ms,
        "flops": flops,
        "mflops": mflops,
        "training_time_seconds": training_time_seconds,
        "parameters_per_ms": num_parameters / max(1, inference_time_ms),
    }
    
    if num_epochs is not None and dataset_size is not None:
        total_training_flops = flops * 3 * num_epochs * dataset_size
        metrics["total_training_flops"] = total_training_flops
        metrics["training_flops_per_second"] = total_training_flops / max(1, training_time_seconds)
    
    return metrics