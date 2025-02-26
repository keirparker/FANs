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
        if torch.cuda.is_available():
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
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("No device specified; using CUDA.")
        else:
            device = torch.device("cpu")
            logger.info("No device specified; using CPU.")

    return device
