#!/usr/bin/env python
"""
Device selection utilities for the ML experimentation framework.
"""

import torch
from loguru import logger


def select_device(config):
    """
    Determine which device to use based on config and hardware availability.

    Args:
        config: Configuration dictionary with hyperparameters

    Returns:
        torch.device: The selected device (cuda, mps, or cpu)
    """
    # Attempt to read device preference from config
    device_str = config["hyperparameters"].get("device", None)

    # Debug CUDA detection issues
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else "unknown"
        logger.info(f"CUDA is available. Found {cuda_device_count} device(s): {cuda_device_name}")
    else:
        logger.warning("CUDA not detected. Debugging information:")
        try:
            import subprocess
            # Try to get nvidia-smi output
            try:
                nvidia_smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode('utf-8')
                logger.info(f"nvidia-smi output:\n{nvidia_smi}")
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("nvidia-smi command failed or not found")
                
            # Check if PyTorch was built with CUDA
            logger.info(f"PyTorch CUDA built: {torch.version.cuda is not None}")
            if torch.version.cuda:
                logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        except Exception as e:
            logger.error(f"Error during CUDA debug: {e}")

    # Force CUDA on p3.8x EC2 if specified in config
    force_cuda = config.get("force_cuda", False)
    if force_cuda and device_str == "cuda":
        logger.info("Forcing CUDA device on p3.8x EC2 instance")
        device = torch.device("cuda")
        return device

    # 1) If user explicitly asked for 'mps' and it's available, use that
    if device_str == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon).")
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    # 2) If user explicitly asked for 'cuda' and it's available, use that
    elif device_str == "cuda":
        if cuda_available:
            device = torch.device("cuda")
            logger.info("Using CUDA device (NVIDIA GPU).")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    # 3) If user explicitly asked for 'cpu', use that
    elif device_str == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU device as requested.")

    # 4) Otherwise, pick the best available automatically
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("No device specified; using MPS for Apple Silicon.")
        elif cuda_available:
            device = torch.device("cuda")
            logger.info("No device specified; using CUDA.")
        else:
            device = torch.device("cpu")
            logger.info("No device specified; using CPU.")

    return device
