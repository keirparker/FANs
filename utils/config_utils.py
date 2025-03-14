#!/usr/bin/env python
"""
Configuration utilities for the ML experimentation framework.
"""

import yaml
import os
from loguru import logger
import random
import numpy as np
import torch



def load_config(config_path="configs/config.yml"):
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file is invalid
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate essential configuration keys
        required_keys = [
            "models_to_run",
            "datasets_to_run",
            "data_versions",
            "hyperparameters",
        ]
        for key in required_keys:
            if key not in config:
                logger.warning(f"Missing '{key}' in configuration file.")

        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def setup_environment(config):
    """
    Setup environment based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        None
    """
    # Create necessary directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Configure logger
    setup_logging(config)

    # Set random seeds for reproducibility
    if "random_seed" in config:

        seed = config["random_seed"]
        
        # 1. Set basic random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 2. Set Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # 3. Set CUDA seeds and deterministic settings
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # For CUDA >= 10.2
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # 4. Limit CPU threads for deterministic behavior
        torch.set_num_threads(1)
        
        # 5. Use deterministic algorithms where possible
        try:
            # For PyTorch 1.8+
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Fallback for older PyTorch
            try:
                torch.set_deterministic(True)
            except AttributeError:
                logger.warning("Could not set deterministic algorithms in PyTorch")
        
        logger.info(f"Random seed set to {seed} with full deterministic configuration enabled")


def setup_logging(config):
    """
    Configure logging based on the configuration.

    Args:
        config: Configuration dictionary

    Returns:
        None
    """
    log_level = config.get("log_level", "INFO").upper()
    log_file = config.get("log_file", "logs/experiment.log")

    # Configure logger
    logger.remove()  # Remove default handlers
    logger.add(log_file, level=log_level, rotation="10 MB")  # File handler
    logger.add(lambda msg: print(msg), level=log_level)  # Console handler

    logger.info(f"Logging configured with level {log_level}")
