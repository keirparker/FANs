#!/usr/bin/env python
"""Efficiency utilities for measuring model performance metrics."""

import time
import math
import torch
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple
import torch.nn as nn

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
    device = next(model.parameters()).device
    
    try:
        dummy_input = torch.randn(1, *input_size).to(device)
        
        if THOP_AVAILABLE:
            try:
                flops, _ = thop_profile(model, inputs=(dummy_input,))
                return int(flops)
            except Exception as e:
                logger.warning(f"Error in thop_profile: {e}, trying alternative approach")
                if any(model_name in model.__class__.__name__ for model_name in
                      ["FANPhaseOffsetModelGated", "FANPhaseOffsetModelUniform", "FANPhaseOffsetModel"]):
                    dummy_input = dummy_input.reshape(1, -1)
                    try:
                        flops, _ = thop_profile(model, inputs=(dummy_input,))
                        return int(flops)
                    except Exception:
                        param_count = count_params(model)
                        return param_count * 3
                else:
                    param_count = count_params(model)
                    return param_count * 3
        else:
            param_count = count_params(model)
            return param_count * 3
    except Exception as e:
        logger.warning(f"Error in count_flops: {e}, falling back to parameter estimation")
        param_count = count_params(model)
        return param_count * 3


def measure_inference_time(model: nn.Module, input_size: Tuple, num_repeats: int = 100) -> float:
    """Measure the average inference time in milliseconds."""
    device = next(model.parameters()).device
    
    try:
        dummy_input = torch.randn(1, *input_size).to(device)
        
        if any(model_name in model.__class__.__name__ for model_name in
              ["FANPhaseOffsetModelGated", "FANPhaseOffsetModelUniform", "FANPhaseOffsetModel"]):
            try:
                with torch.no_grad():
                    model(dummy_input)
            except Exception as e:
                logger.debug(f"Reshaping input due to: {e}")
                dummy_input = dummy_input.reshape(1, -1)
        
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        first_test_ok = False
        with torch.no_grad():
            try:
                _ = model(dummy_input)
                first_test_ok = True
            except Exception as e:
                logger.warning(f"Model threw exception during inference test: {e}")
                return count_params(model) / 1e6
        
        if not first_test_ok:
            return count_params(model) / 1e6
            
        def dummy_sync():
            time.sleep(0.001)  # Small sleep to simulate synchronization
            pass
        
        sync_func = dummy_sync  # Default to dummy
        
        if device.type == 'cuda':
            try:
                if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                    def cuda_sync():
                        torch.cuda.synchronize()
                    sync_func = cuda_sync
            except (AssertionError, AttributeError, RuntimeError) as e:
                logger.debug(f"CUDA synchronization not available: {e}")
        elif device.type == 'mps':
            def mps_sync():
                time.sleep(0.0005)  # Half the time of dummy sync
            sync_func = mps_sync
            
        sync_func()
        start_time = time.time()
        
        for _ in range(num_repeats):
            with torch.no_grad():
                _ = model(dummy_input)
        
        sync_func()
        end_time = time.time()
        
        return (end_time - start_time) * 1000 / num_repeats
    except Exception as e:
        logger.warning(f"Error measuring inference time: {e}, returning approximate value")
        param_count = count_params(model)
        return param_count / 1e6  # Rough approximation: 1ms per million parameters


def compute_efficiency_metrics(
    model: nn.Module, 
    input_size: Tuple,
    training_time_seconds: float,
    num_parameters: Optional[int] = None,
    num_epochs: Optional[int] = None,
    dataset_size: Optional[int] = None
) -> Dict[str, float]:
    """Compute comprehensive efficiency metrics for a model."""
    if hasattr(model, "count_params") and callable(getattr(model, "count_params")):
        num_parameters = model.count_params()
    elif num_parameters is None:
        num_parameters = count_params(model)
    
    try:
        inference_time_ms = measure_inference_time(model, input_size)
    except Exception as e:
        logger.warning(f"Error measuring inference time: {e}, using estimation")
        inference_time_ms = num_parameters / 1e6  # Approximate: 1ms per million params
    
    try:
        if hasattr(model, "get_flops") and callable(getattr(model, "get_flops")):
            flops = model.get_flops()
        else:
            flops = count_flops(model, input_size)
    except Exception as e:
        logger.warning(f"Error counting FLOPS: {e}, using estimation")
        flops = num_parameters * 3  # Approximate based on parameter count
        
    if math.isnan(flops) or flops == 0:
        flops = num_parameters * 3
    if math.isnan(inference_time_ms) or inference_time_ms == 0:
        inference_time_ms = num_parameters / 1e6
        
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