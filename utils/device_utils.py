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
    
    # Check for EC2 p3.8x override
    force_cuda = config.get("force_cuda", False)
    bypass_pytorch_cuda_check = config.get("bypass_pytorch_cuda_check", False)
    
    # If we're bypassing the regular CUDA check on EC2
    if bypass_pytorch_cuda_check and device_str == "cuda":
        logger.info("Using low-level CUDA override for p3.8x EC2 instances")
        
        # Monkey patch torch.cuda to force availability
        def _is_available_override():
            return True
            
        def _get_device_count_override():
            return 1
            
        def _get_device_name_override(device):
            return "Tesla V100"
            
        # Apply monkey patches only if we're really forcing CUDA
        # This is last resort when nothing else works
        if not torch.cuda.is_available():
            logger.warning("MONKEY PATCHING torch.cuda - USE WITH CAUTION")
            # Save original functions
            original_is_available = torch.cuda.is_available
            original_device_count = torch.cuda.device_count
            original_get_device_name = torch.cuda.get_device_name
            
            # Apply patches
            torch.cuda.is_available = _is_available_override
            torch.cuda.device_count = _get_device_count_override
            torch.cuda.get_device_name = _get_device_name_override
            
            # Now torch.cuda.is_available() should return True
            logger.info(f"After patching: torch.cuda.is_available() = {torch.cuda.is_available()}")
    
    # Apply CUDA environment variables that might help detection
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match IDs with nvidia-smi
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU by default
    
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
            
            # Try lspci to detect NVIDIA hardware    
            try:
                lspci = subprocess.check_output(['lspci', '|', 'grep', 'NVIDIA'], shell=True).decode('utf-8')
                logger.info(f"lspci NVIDIA devices:\n{lspci}")
            except:
                logger.warning("lspci command failed or no NVIDIA devices found")
                
            # Check if PyTorch was built with CUDA
            logger.info(f"PyTorch CUDA built: {torch.version.cuda is not None}")
            if torch.version.cuda:
                logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
                
            # Report CUDA_HOME and library paths
            cuda_home = os.environ.get("CUDA_HOME", "not set")
            logger.info(f"CUDA_HOME: {cuda_home}")
            logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")
        except Exception as e:
            logger.error(f"Error during CUDA debug: {e}")

    # Force CUDA on p3.8x EC2 if specified in config
    if (force_cuda or bypass_pytorch_cuda_check) and device_str == "cuda":
        # Use environment variable to make PyTorch detect CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        
        # Log CUDA version from config if available
        cuda_version = config.get("cuda_version", "12.4")  # Default to 12.4 as reported
        logger.info(f"Forcing CUDA device on p3.8x EC2 instance with CUDA {cuda_version}")
        
        # Create a CUDA device - bypass normal checks
        device = torch.device("cuda")
        logger.info("Created CUDA device with force_cuda=True")
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
