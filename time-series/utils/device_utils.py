#!/usr/bin/env python
"""
Device selection utilities for the ML experimentation framework.
Provides comprehensive support for different hardware platforms:
- Apple Silicon (M1/M2/M3) with MPS acceleration
- NVIDIA GPUs with CUDA
- AWS EC2 instances (specifically p3dn.24xlarge with 8x Tesla V100)
- Windows machines with Gigabyte GPUs
- CPU fallback
"""

import torch
import os
import platform
import subprocess
import re
from loguru import logger


def detect_aws_instance_type():
    """
    Detect if running on an AWS EC2 instance and identify the instance type.
    
    Returns:
        str or None: AWS EC2 instance type if detected, None otherwise
    """
    instance_type = None
    
    try:
        import urllib.request
        import socket
        socket.setdefaulttimeout(1)
        response = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-type')
        instance_type = response.read().decode('utf-8')
        logger.info(f"Detected AWS EC2 instance type: {instance_type}")
        return instance_type
    except:
        try:
            if platform.system() == 'Linux':
                try:
                    dmi = subprocess.check_output(['cat', '/sys/devices/virtual/dmi/id/sys_vendor']).decode('utf-8').strip()
                    if 'Amazon' in dmi:
                        product = subprocess.check_output(['cat', '/sys/devices/virtual/dmi/id/product_name']).decode('utf-8').strip()
                        if product:
                            logger.info(f"Detected AWS EC2 instance from DMI: {product}")
                            return product
                except:
                    pass
                    
                try:
                    nvidia_smi = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')
                    gpus = nvidia_smi.strip().split('\n')
                    
                    v100_count = sum(1 for gpu in gpus if 'Tesla V100' in gpu)
                    
                    if v100_count == 8:
                        logger.info("Detected 8x Tesla V100 GPUs, likely p3dn.24xlarge")
                        return "p3dn.24xlarge"
                    elif v100_count == 4:
                        logger.info("Detected 4x Tesla V100 GPUs, likely p3.8xlarge")
                        return "p3.8xlarge"
                    elif v100_count > 0:
                        logger.info(f"Detected {v100_count}x Tesla V100 GPUs, likely p3 family")
                        return f"p3-{v100_count}gpu"
                except:
                    pass
        except:
            pass
    
    return None


def detect_windows_gigabyte_gpu():
    """
    Detect if running on a Windows system with Gigabyte GPU.
    
    Returns:
        bool: True if running on Windows with Gigabyte GPU, False otherwise
        dict: Information about the detected Gigabyte GPU
    """
    is_windows_gigabyte = False
    gigabyte_info = {
        'model': None,
        'vram': None,
        'driver_version': None,
        'optimized': False
    }
    
    if platform.system() == 'Windows':
        if torch.cuda.is_available():
            try:
                nvidia_smi = subprocess.check_output(['nvidia-smi', '-q'], stderr=subprocess.DEVNULL).decode('utf-8')
                
                is_nvidia_gpu = any(brand in nvidia_smi.upper() for brand in ['NVIDIA', 'GEFORCE', 'GTX', 'RTX'])
                if is_nvidia_gpu:
                    is_windows_gigabyte = True  # Use this flag for all NVIDIA GPUs
                    logger.info("Detected NVIDIA GPU on Windows")
                    
                    model_match = re.search(r'Product Name\s+:\s+(.*)', nvidia_smi)
                    if model_match:
                        gigabyte_info['model'] = model_match.group(1).strip()
                        logger.info(f"NVIDIA GPU model: {gigabyte_info['model']}")
                    
                    vram_match = re.search(r'FB Memory Usage\s+.*?Total\s+:\s+(\d+)\s+MiB', nvidia_smi, re.DOTALL)
                    if vram_match:
                        vram_mb = int(vram_match.group(1))
                        gigabyte_info['vram'] = vram_mb
                        logger.info(f"NVIDIA GPU VRAM: {vram_mb} MB")
                    
                    driver_match = re.search(r'Driver Version\s+:\s+([\d\.]+)', nvidia_smi)
                    if driver_match:
                        gigabyte_info['driver_version'] = driver_match.group(1)
                        logger.info(f"NVIDIA driver version: {gigabyte_info['driver_version']}")
                
                if not is_windows_gigabyte and torch.cuda.device_count() > 0:
                    gpu_name = torch.cuda.get_device_name(0).upper()
                    is_nvidia_gpu = any(brand in gpu_name for brand in ['NVIDIA', 'GEFORCE', 'GTX', 'RTX'])
                    if is_nvidia_gpu or 'GIGABYTE' in gpu_name or 'AORUS' in gpu_name:
                        is_windows_gigabyte = True
                        gigabyte_info['model'] = torch.cuda.get_device_name(0)
                        logger.info(f"Detected GPU via PyTorch: {gigabyte_info['model']}")
                        
                        try:
                            free_mem, total_mem = torch.cuda.mem_get_info(0)
                            gigabyte_info['vram'] = int(total_mem / (1024 * 1024))  # Convert to MB
                            logger.info(f"GPU VRAM: {gigabyte_info['vram']} MB")
                        except:
                            pass
            except Exception as e:
                logger.warning(f"Error detecting Gigabyte GPU: {e}")
    
    return is_windows_gigabyte, gigabyte_info


def detect_apple_silicon():
    """
    Detect if running on Apple Silicon (M1/M2/M3) hardware.
    
    Returns:
        bool: True if running on Apple Silicon, False otherwise
        str or None: Apple Silicon chip model if available (M1, M2, M3, etc)
    """
    is_apple_silicon = False
    chip_model = None
    
    if platform.system() == 'Darwin':
        mps_available = torch.backends.mps.is_available()
        
        try:
            chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            
            if 'Apple' in chip_info:
                is_apple_silicon = True
                
                match = re.search(r'Apple\s+(M\d+)', chip_info)
                if match:
                    chip_model = match.group(1)
                    logger.info(f"Detected Apple Silicon: {chip_model}")
                else:
                    logger.info(f"Detected Apple Silicon: {chip_info}")
            elif mps_available:
                is_apple_silicon = True
                logger.info("Detected Apple Silicon via MPS availability")
        except:
            is_apple_silicon = mps_available
            if is_apple_silicon:
                logger.info("Detected Apple Silicon via MPS availability")
    
    return is_apple_silicon, chip_model


def detect_gpu_capabilities():
    """
    Detect and catalog available GPU capabilities.
    
    Returns:
        dict: Dictionary containing GPU capabilities
    """
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'cuda_devices': [],
        'cuda_version': torch.version.cuda,
        'mps_available': torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        'apple_silicon': False,
        'chip_model': None,
        'total_gpu_memory': 0,
        'is_aws': False,
        'aws_instance_type': None,
        'is_windows': platform.system() == 'Windows',
        'is_windows_gigabyte': False,
        'gigabyte_gpu_info': None,
    }
    
    gpu_info['apple_silicon'], gpu_info['chip_model'] = detect_apple_silicon()
    
    aws_instance = detect_aws_instance_type()
    if aws_instance:
        gpu_info['is_aws'] = True
        gpu_info['aws_instance_type'] = aws_instance
        
    if gpu_info['is_windows']:
        is_windows_gigabyte, gigabyte_gpu_info = detect_windows_gigabyte_gpu()
        gpu_info['is_windows_gigabyte'] = is_windows_gigabyte
        gpu_info['gigabyte_gpu_info'] = gigabyte_gpu_info
    
    if gpu_info['cuda_available']:
        try:
            gpu_info['cuda_device_count'] = torch.cuda.device_count()
            
            if gpu_info['cuda_device_count'] == 0 or gpu_info['cuda_device_count'] == 1:
                try:
                    import subprocess
                    nvidia_smi = subprocess.check_output(['nvidia-smi', '-L'], stderr=subprocess.DEVNULL).decode('utf-8')
                    gpu_lines = nvidia_smi.strip().split('\n')
                    
                    if len(gpu_lines) > gpu_info['cuda_device_count']:
                        logger.warning(f"PyTorch only detected {gpu_info['cuda_device_count']} GPUs but nvidia-smi shows {len(gpu_lines)}")
                        gpu_info['cuda_device_count'] = len(gpu_lines)
                        
                        if any('Tesla V100' in line for line in gpu_lines) and len(gpu_lines) >= 4:
                            logger.info("Detected Tesla V100 GPUs, most likely on p3dn.24xlarge with 8 GPUs")
                            gpu_info['cuda_device_count'] = 8
                except Exception as e:
                    logger.warning(f"Failed to get GPU count from nvidia-smi: {e}")
            
            if gpu_info['is_aws'] and gpu_info['aws_instance_type'] == 'p3dn.24xlarge':
                logger.info("Creating device info for all 8 GPUs on p3dn.24xlarge")
                for i in range(8):
                    device_props = {
                        'index': i,
                        'name': "Tesla V100-SXM2-32GB",
                        'compute_capability': "7.0",
                        'total_memory': 32 * 1024 * 1024 * 1024,  # 32GB in bytes
                    }
                    gpu_info['cuda_devices'].append(device_props)
                    gpu_info['total_gpu_memory'] += device_props['total_memory']
                
                return gpu_info
            
            current_device = torch.cuda.current_device()
            for i in range(gpu_info['cuda_device_count']):
                try:
                    torch.cuda.set_device(i)
                    device_name = torch.cuda.get_device_name(i)
                    
                    if i == current_device:
                        try:
                            device_cap = torch.cuda.get_device_capability(i)
                            compute_capability = f"{device_cap[0]}.{device_cap[1]}"
                        except:
                            compute_capability = "unknown"
                    else:
                        compute_capability = "unknown"
                        
                    device_props = {
                        'index': i,
                        'name': device_name,
                        'compute_capability': compute_capability,
                    }
                    
                    if i == current_device:
                        try:
                            free_mem, total_mem = torch.cuda.mem_get_info(i)
                            device_props['total_memory'] = total_mem
                            device_props['free_memory'] = free_mem
                            gpu_info['total_gpu_memory'] += total_mem
                        except:
                            pass
                    
                    gpu_info['cuda_devices'].append(device_props)
                except Exception as e:
                    logger.warning(f"Error getting detailed info for CUDA device {i}: {e}")
                    device_props = {
                        'index': i,
                        'name': f"GPU {i}",
                        'compute_capability': "unknown",
                    }
                    gpu_info['cuda_devices'].append(device_props)
                
            torch.cuda.set_device(current_device)
            
        except Exception as e:
            logger.warning(f"Error gathering CUDA device information: {e}")
    
    return gpu_info


def setup_aws_p3dn_environment(conservative=True):
    """
    Configure environment variables for optimal performance on AWS p3dn.24xlarge instances.
    These settings are specifically tuned for instances with 8x Tesla V100 GPUs and EFA networking.
    
    Args:
        conservative (bool): If True, initially use only a single GPU to avoid CUDA errors,
                           GPUs will be expanded later in training_utils.py
    """
    os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debug info
    os.environ["NCCL_IB_DISABLE"] = "0"  # Enable InfiniBand for interconnect
    os.environ["NCCL_IB_GID_INDEX"] = "3"  # Optimal for AWS Elastic Fabric Adapter (EFA)
    os.environ["NCCL_IB_HCA"] = "^mlx5_0"  # Use Mellanox adapters
    os.environ["NCCL_IB_TC"] = "106"  # Traffic class for IB
    os.environ["NCCL_IB_TIMEOUT"] = "23"  # Longer timeout for stability
    os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"  # Skip loopback, docker interfaces
    os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable GPU P2P
    os.environ["NCCL_P2P_LEVEL"] = "NVL"  # NVLink for P2P
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match IDs with nvidia-smi
    
    if conservative:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Start with just one GPU
        logger.info("Conservative mode: starting with only GPU 0")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(8))
        logger.info("Performance mode: using all 8 GPUs")
    
    os.environ["CUDA_CACHE_DISABLE"] = "0"  # Enable JIT cache
    os.environ["CUDA_AUTO_BOOST"] = "0"  # Disable autoboost for consistent performance
    
    os.environ["OMP_NUM_THREADS"] = "8"  # Limit OpenMP threads
    os.environ["KMP_BLOCKTIME"] = "0"  # Don't let OpenMP worker threads sleep
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"  # CPU affinity
    
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # Debug info for distributed
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"  # More verbose logging
    
    try:
        import torch
        torch.cuda.init()
        torch.cuda.set_device(0)
    except:
        logger.warning("Could not initialize CUDA directly")
    
    logger.info("Configured environment variables for AWS p3dn.24xlarge")


def setup_windows_gigabyte_environment(gigabyte_info):
    """
    Configure environment variables for optimal performance on Windows with Gigabyte GPUs.
    
    Args:
        gigabyte_info: Dictionary with Gigabyte GPU information
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match IDs with nvidia-smi
    os.environ["CUDA_CACHE_DISABLE"] = "0"  # Enable JIT cache
    
    num_cpu_cores = os.cpu_count() or 4
    os.environ["OMP_NUM_THREADS"] = str(min(num_cpu_cores, 8))  # Limit OpenMP threads to avoid oversubscription
    os.environ["MKL_NUM_THREADS"] = str(min(num_cpu_cores, 8))  # Limit MKL threads similarly
    
    if gigabyte_info and gigabyte_info.get('vram'):
        vram_mb = gigabyte_info.get('vram')
        if vram_mb:
            if vram_mb >= 8000:  # 8GB or more
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    os.environ["CUDNN_LOGINFO_DBG"] = "0"  # Disable verbose logging
    os.environ["CUDNN_LOGDEST_DBG"] = "stdout"  # Log to stdout for better diagnostics if needed
    
    logger.info("Configured environment variables for Windows with Gigabyte GPU")
    gigabyte_info['optimized'] = True


def setup_apple_silicon_environment():
    """
    Configure environment variables for optimal performance on Apple Silicon (M1/M2/M3) hardware.
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
    
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # Use all cores for OpenMP
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())  # Use all cores for MKL
    
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Don't limit MPS memory
    
    logger.info("Configured environment variables for Apple Silicon")


def select_device(config):
    """
    Enhanced device selection with platform-specific optimizations. 
    Automatically detects and configures for:
    - AWS p3dn.24xlarge instances with 8x Tesla V100 GPUs
    - Apple Silicon (M1/M2/M3) with MPS acceleration
    - Any CUDA-compatible GPU
    - CPU fallback

    Args:
        config: Configuration dictionary with hyperparameters

    Returns:
        torch.device: The selected device (cuda, mps, or cpu)
        dict: Additional information about the device configuration
    """
    device_str = config["hyperparameters"].get("device", None)
    force_cuda = config.get("force_cuda", False)
    bypass_pytorch_cuda_check = config.get("bypass_pytorch_cuda_check", False)
    
    gpu_info = detect_gpu_capabilities()
    
    logger.info("Detected system capabilities:")
    logger.info(f"  CUDA: {'Available' if gpu_info['cuda_available'] else 'Not available'}")
    if gpu_info['cuda_available']:
        logger.info(f"  CUDA version: {gpu_info['cuda_version']}")
        logger.info(f"  CUDA devices: {gpu_info['cuda_device_count']}")
        for device in gpu_info['cuda_devices']:
            logger.info(f"    Device {device['index']}: {device['name']}")
    
    logger.info(f"  MPS: {'Available' if gpu_info['mps_available'] else 'Not available'}")
    if gpu_info['apple_silicon']:
        logger.info(f"  Apple Silicon: {gpu_info['chip_model'] or 'Yes'}")
    
    if gpu_info['is_aws']:
        logger.info(f"  AWS EC2 instance: {gpu_info['aws_instance_type']}")
        
    if gpu_info['is_windows']:
        logger.info(f"  Windows: {platform.release()}")
        if gpu_info['is_windows_gigabyte']:
            gigabyte_info = gpu_info['gigabyte_gpu_info']
            logger.info(f"  Windows Gigabyte GPU: {gigabyte_info.get('model', 'Unknown model')}")
            if gigabyte_info.get('vram'):
                logger.info(f"  GPU VRAM: {gigabyte_info.get('vram')} MB")
            if gigabyte_info.get('driver_version'):
                logger.info(f"  NVIDIA driver: {gigabyte_info.get('driver_version')}")
    
    if gpu_info['is_aws'] and gpu_info['aws_instance_type'] == 'p3dn.24xlarge':
        logger.info("Optimizing for AWS p3dn.24xlarge instance")
        setup_aws_p3dn_environment(conservative=True)  # Start with conservative mode for stability
        
        aws_p3dn_optimization = config.get("hyperparameters", {}).get("aws_p3dn_optimization", True)
        
        aws_max_gpus = config.get("hyperparameters", {}).get("aws_max_gpus", 8)
        if aws_max_gpus < 8:
            import os
            visible_devices = ",".join(str(i) for i in range(aws_max_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            logger.info(f"Limiting to {aws_max_gpus} GPUs as specified in config: CUDA_VISIBLE_DEVICES={visible_devices}")
        
        device_str = "cuda"
        force_cuda = aws_p3dn_optimization  # Only force if optimization is enabled
        
    elif gpu_info['is_windows_gigabyte']:
        logger.info(f"Optimizing for Windows with Gigabyte GPU: {gpu_info['gigabyte_gpu_info'].get('model', 'Unknown model')}")
        setup_windows_gigabyte_environment(gpu_info['gigabyte_gpu_info'])
        
        windows_gigabyte_optimization = config.get("hyperparameters", {}).get("windows_gigabyte_optimization", True)
        
        device_str = "cuda"
        force_cuda = windows_gigabyte_optimization
        
    elif gpu_info['apple_silicon']:
        logger.info(f"Optimizing for Apple Silicon: {gpu_info['chip_model'] or 'Unknown model'}")
        setup_apple_silicon_environment()
        
        if not device_str:
            device_str = "mps"
    
    if force_cuda and (not gpu_info['cuda_available'] or bypass_pytorch_cuda_check):
        logger.info("Force CUDA requested, applying advanced detection methods")
        
        try:
            nvidia_smi = subprocess.check_output(['nvidia-smi', '-L'], stderr=subprocess.DEVNULL).decode('utf-8')
            if 'Tesla V100' in nvidia_smi:
                logger.info(f"Detected Tesla V100 GPUs from nvidia-smi:\n{nvidia_smi.strip()}")
                
                if bypass_pytorch_cuda_check:
                    logger.warning("MONKEY PATCHING torch.cuda for Tesla V100 detection - USE WITH CAUTION")
                    
                    original_is_available = torch.cuda.is_available
                    original_device_count = torch.cuda.device_count
                    original_get_device_name = torch.cuda.get_device_name
                    
                    gpu_count = len(nvidia_smi.strip().split('\n'))
                    
                    original_is_available = torch.cuda.is_available
                    original_device_count = torch.cuda.device_count
                    original_get_device_name = torch.cuda.get_device_name
                    
                    def _is_available_override():
                        return True
                        
                    def _get_device_count_override():
                        return gpu_count
                        
                    def _get_device_name_override(device):
                        return "Tesla V100"
                    
                    torch.cuda.is_available = _is_available_override
                    torch.cuda.device_count = _get_device_count_override
                    torch.cuda.get_device_name = _get_device_name_override
                    
                    gpu_info['cuda_available'] = True
                    gpu_info['cuda_device_count'] = gpu_count
                    
                    logger.info(f"After patching: CUDA available: {torch.cuda.is_available()}")
                    logger.info(f"After patching: CUDA device count: {torch.cuda.device_count()}")
                    
                    import atexit
                    def _restore_cuda_functions():
                        logger.info("Restoring original CUDA functions")
                        torch.cuda.is_available = original_is_available
                        torch.cuda.device_count = original_device_count
                        torch.cuda.get_device_name = original_get_device_name
                    
                    atexit.register(_restore_cuda_functions)
        except Exception as e:
            logger.warning(f"Advanced CUDA detection failed: {e}")
    
    device_info = {
        'type': None,
        'name': None,
        'is_multi_gpu': False,
        'gpu_count': 0,
        'platform': platform.system(),
        'platform_specific': gpu_info['apple_silicon'] or gpu_info['is_aws'] or gpu_info['is_windows_gigabyte'],
        'aws_instance': gpu_info['aws_instance_type'] if gpu_info['is_aws'] else None,
        'apple_silicon': gpu_info['apple_silicon'],
        'chip_model': gpu_info['chip_model'],
        'is_windows': gpu_info['is_windows'],
        'is_windows_gigabyte': gpu_info['is_windows_gigabyte'],
        'gigabyte_gpu_info': gpu_info['gigabyte_gpu_info'] if gpu_info['is_windows_gigabyte'] else None,
    }
    
    if device_str == "cuda" and (gpu_info['cuda_available'] or force_cuda):
        device = torch.device("cuda")
        device_info['type'] = 'cuda'
        device_info['gpu_count'] = gpu_info['cuda_device_count']
        device_info['is_multi_gpu'] = device_info['gpu_count'] > 1
        
        if device_info['gpu_count'] > 0:
            device_info['name'] = torch.cuda.get_device_name(0)
        
        logger.info(f"Using CUDA with {device_info['gpu_count']} device(s)")
        if device_info['is_multi_gpu']:
            logger.info(f"Multi-GPU training enabled with {device_info['gpu_count']} GPUs")
        
    elif device_str == "mps" and gpu_info['mps_available']:
        device = torch.device("mps")
        device_info['type'] = 'mps'
        device_info['name'] = f"Apple {gpu_info['chip_model'] or 'Silicon'}"
        logger.info(f"Using MPS for Apple Silicon: {device_info['name']}")
        
    elif device_str == "cpu" or (not device_str and not gpu_info['cuda_available'] and not gpu_info['mps_available']):
        device = torch.device("cpu")
        device_info['type'] = 'cpu'
        device_info['name'] = platform.processor() or "CPU"
        logger.info(f"Using CPU: {device_info['name']}")
        
    else:
        if gpu_info['cuda_available']:
            device = torch.device("cuda")
            device_info['type'] = 'cuda'
            device_info['gpu_count'] = gpu_info['cuda_device_count']
            device_info['is_multi_gpu'] = device_info['gpu_count'] > 1
            
            if device_info['gpu_count'] > 0:
                device_info['name'] = torch.cuda.get_device_name(0)
                
            logger.info(f"Auto-selected CUDA with {device_info['gpu_count']} device(s)")
            
        elif gpu_info['mps_available']:
            device = torch.device("mps")
            device_info['type'] = 'mps'
            device_info['name'] = f"Apple {gpu_info['chip_model'] or 'Silicon'}"
            logger.info(f"Auto-selected MPS for Apple Silicon: {device_info['name']}")
            
        else:
            device = torch.device("cpu")
            device_info['type'] = 'cpu'
            device_info['name'] = platform.processor() or "CPU"
            logger.info(f"Auto-selected CPU: {device_info['name']}")
    
    return device, device_info
