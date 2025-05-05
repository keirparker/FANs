#!/usr/bin/env python
"""
Configuration utilities for the ML experimentation framework.
"""

import yaml
import os
import platform
import multiprocessing

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

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
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(current_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)

    setup_logging(config)

    system_platform = platform.system()
    if system_platform == 'Windows':
        logger.info(f"Detected Windows platform: {platform.release()}")
        # Windows-specific setup
        setup_windows_environment(config)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        
        if config.get("hyperparameters", {}).get("aws_max_gpus", 0) > 0:
            max_gpus = config["hyperparameters"]["aws_max_gpus"]
            visible_devices = ",".join(str(i) for i in range(min(max_gpus, torch.cuda.device_count())))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            logger.info(f"Limiting to {max_gpus} GPUs: CUDA_VISIBLE_DEVICES={visible_devices}")

    performance_setting = config.get("performance_setting", "fast")
    performance_mode = performance_setting.lower() == "fast"


def setup_windows_environment(config):
    """Configure specific settings for Windows environments"""
    logger.info("Setting up Windows-specific environment...")

    performance_setting = config.get("performance_setting", "fast")
    performance_mode = performance_setting.lower() == "fast"

    os.environ["PYTHONIOENCODING"] = "utf-8"

    if os.cpu_count():
        suggested_threads = min(os.cpu_count(), 8)
        os.environ["OMP_NUM_THREADS"] = str(suggested_threads)
        os.environ["MKL_NUM_THREADS"] = str(suggested_threads)
        logger.info(f"Set Windows thread count to {suggested_threads}")

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_CACHE_DISABLE"] = "0"
        
        if torch.cuda.device_count() > 0:
            try:
                gpu_name = torch.cuda.get_device_name(0).upper()
                logger.info(f"Detected CUDA GPU: {gpu_name}")
                is_nvidia_gpu = any(brand in gpu_name for brand in ['NVIDIA', 'GEFORCE', 'GTX', 'RTX'])
                
                if is_nvidia_gpu or 'GIGABYTE' in gpu_name or 'AORUS' in gpu_name:
                    logger.info(f"Optimizing for GPU on Windows: {gpu_name}")
                    if "device_info" not in config:
                        config["device_info"] = {}
                    config["device_info"]["platform"] = "Windows"
                    config["device_info"]["is_windows"] = True
                    config["device_info"]["is_windows_gigabyte"] = True
                    
                    try:
                        free_mem, total_mem = torch.cuda.mem_get_info(0)
                        vram_mb = int(total_mem / (1024 * 1024))
                        logger.info(f"GPU VRAM: {vram_mb} MB")
                        
                        if "gigabyte_gpu_info" not in config["device_info"]:
                            config["device_info"]["gigabyte_gpu_info"] = {}
                        config["device_info"]["gigabyte_gpu_info"]["vram"] = vram_mb
                    except:
                        logger.info("Could not get VRAM information")
                    
                    if config.get("hyperparameters", {}).get("windows_memory_optimization", True):
                        torch._C._jit_set_profiling_executor(True)
                        torch._C._jit_set_profiling_mode(True)
                        logger.info("Enabled PyTorch JIT optimizations for Windows GPU")
                        
                        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                        logger.info("Set CUDA memory allocation configuration for better efficiency")
            except Exception as e:
                logger.warning(f"Error detecting GPU information on Windows: {e}")
    
    logger.info("Windows environment setup complete")

    if "random_seed" in config:
        seed = config["random_seed"]
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            if performance_mode:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16384:8"
                import multiprocessing
                num_threads = multiprocessing.cpu_count()
                torch.set_num_threads(num_threads)
                
                try:
                    torch.use_deterministic_algorithms(False)
                except AttributeError:
                    pass
                
                logger.info("Performance mode enabled for Tesla V100 GPUs")
            else:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.set_num_threads(1)
                
                try:
                    torch.use_deterministic_algorithms(True)
                except AttributeError:
                    try:
                        torch.set_deterministic(True)
                    except AttributeError:
                        logger.warning("Could not set deterministic algorithms in PyTorch")
        
        logger.info(f"Random seed set to {seed} with performance_setting='{performance_setting}' mode enabled")


def setup_logging(config):
    """Configure logging based on the configuration."""
    log_level = config.get("log_level", "INFO").upper()

    time_series_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log_file_name = config.get("log_file", "experiment.log")
    log_file = os.path.join(time_series_dir, "logs", os.path.basename(log_file_name))

    logger.remove()
    logger.add(log_file, level=log_level, rotation="10 MB")
    logger.add(lambda msg: print(msg), level=log_level)

    logger.info(f"Logging configured with level {log_level} to {log_file}")