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
    
    # Try to get instance info from EC2 metadata service
    try:
        # Use a short timeout for the metadata service
        import urllib.request
        import socket
        socket.setdefaulttimeout(1)
        response = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-type')
        instance_type = response.read().decode('utf-8')
        logger.info(f"Detected AWS EC2 instance type: {instance_type}")
        return instance_type
    except:
        # Try to get from system info
        try:
            # On Linux, check for AWS-specific info
            if platform.system() == 'Linux':
                # Check if DMI info contains Amazon
                try:
                    dmi = subprocess.check_output(['cat', '/sys/devices/virtual/dmi/id/sys_vendor']).decode('utf-8').strip()
                    if 'Amazon' in dmi:
                        # Try to get instance type from product name
                        product = subprocess.check_output(['cat', '/sys/devices/virtual/dmi/id/product_name']).decode('utf-8').strip()
                        if product:
                            logger.info(f"Detected AWS EC2 instance from DMI: {product}")
                            return product
                except:
                    pass
                    
                # Try to infer from GPU count and type
                try:
                    nvidia_smi = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')
                    gpus = nvidia_smi.strip().split('\n')
                    
                    # Count Tesla V100 GPUs
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
    
    # Check if running on Windows
    if platform.system() == 'Windows':
        # First check if CUDA is available at all
        if torch.cuda.is_available():
            try:
                # Try to get nvidia-smi output
                nvidia_smi = subprocess.check_output(['nvidia-smi', '-q'], stderr=subprocess.DEVNULL).decode('utf-8')
                
                # Check GPU vendor information
                # Gigabyte GPUs will show the model name in nvidia-smi output
                if 'GIGABYTE' in nvidia_smi.upper() or 'AORUS' in nvidia_smi.upper():
                    is_windows_gigabyte = True
                    logger.info("Detected Gigabyte GPU on Windows")
                    
                    # Extract GPU model
                    model_match = re.search(r'Product Name\s+:\s+(.*)', nvidia_smi)
                    if model_match:
                        gigabyte_info['model'] = model_match.group(1).strip()
                        logger.info(f"Gigabyte GPU model: {gigabyte_info['model']}")
                    
                    # Extract VRAM size
                    vram_match = re.search(r'FB Memory Usage\s+.*?Total\s+:\s+(\d+)\s+MiB', nvidia_smi, re.DOTALL)
                    if vram_match:
                        vram_mb = int(vram_match.group(1))
                        gigabyte_info['vram'] = vram_mb
                        logger.info(f"Gigabyte GPU VRAM: {vram_mb} MB")
                    
                    # Extract driver version
                    driver_match = re.search(r'Driver Version\s+:\s+([\d\.]+)', nvidia_smi)
                    if driver_match:
                        gigabyte_info['driver_version'] = driver_match.group(1)
                        logger.info(f"NVIDIA driver version: {gigabyte_info['driver_version']}")
                
                # Alternative method: get GPU name directly from PyTorch
                if not is_windows_gigabyte and torch.cuda.device_count() > 0:
                    gpu_name = torch.cuda.get_device_name(0).upper()
                    if 'GIGABYTE' in gpu_name or 'AORUS' in gpu_name:
                        is_windows_gigabyte = True
                        gigabyte_info['model'] = torch.cuda.get_device_name(0)
                        logger.info(f"Detected Gigabyte GPU via PyTorch: {gigabyte_info['model']}")
                        
                        # Try to get memory info
                        try:
                            free_mem, total_mem = torch.cuda.mem_get_info(0)
                            gigabyte_info['vram'] = int(total_mem / (1024 * 1024))  # Convert to MB
                            logger.info(f"Gigabyte GPU VRAM: {gigabyte_info['vram']} MB")
                        except:
                            # Older PyTorch versions might not have mem_get_info
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
    
    # Check if running on macOS
    if platform.system() == 'Darwin':
        # Check if MPS is available - reliable indicator for Apple Silicon
        mps_available = torch.backends.mps.is_available()
        
        # Try to get more detailed chip info
        try:
            # Use sysctl to get chip info on macOS
            chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            
            # Extract chip model from output
            if 'Apple' in chip_info:
                is_apple_silicon = True
                
                # Try to extract the specific M1/M2/M3 model
                match = re.search(r'Apple\s+(M\d+)', chip_info)
                if match:
                    chip_model = match.group(1)
                    logger.info(f"Detected Apple Silicon: {chip_model}")
                else:
                    logger.info(f"Detected Apple Silicon: {chip_info}")
            elif mps_available:
                # If we can't identify specific chip but MPS is available,
                # it's definitely Apple Silicon
                is_apple_silicon = True
                logger.info("Detected Apple Silicon via MPS availability")
        except:
            # Fallback detection using MPS
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
    
    # Check if we're on Apple Silicon
    gpu_info['apple_silicon'], gpu_info['chip_model'] = detect_apple_silicon()
    
    # Check if we're on AWS
    aws_instance = detect_aws_instance_type()
    if aws_instance:
        gpu_info['is_aws'] = True
        gpu_info['aws_instance_type'] = aws_instance
        
    # Check if we're on Windows with a Gigabyte GPU
    if gpu_info['is_windows']:
        is_windows_gigabyte, gigabyte_gpu_info = detect_windows_gigabyte_gpu()
        gpu_info['is_windows_gigabyte'] = is_windows_gigabyte
        gpu_info['gigabyte_gpu_info'] = gigabyte_gpu_info
    
    # Gather CUDA device information if available
    if gpu_info['cuda_available']:
        try:
            gpu_info['cuda_device_count'] = torch.cuda.device_count()
            
            # For AWS p3dn.24xlarge, we might need to check nvidia-smi directly
            if gpu_info['cuda_device_count'] == 0 or gpu_info['cuda_device_count'] == 1:
                try:
                    # Try to get device count from nvidia-smi
                    import subprocess
                    nvidia_smi = subprocess.check_output(['nvidia-smi', '-L'], stderr=subprocess.DEVNULL).decode('utf-8')
                    gpu_lines = nvidia_smi.strip().split('\n')
                    
                    # If we found multiple GPUs but PyTorch only sees 1, update the count
                    if len(gpu_lines) > gpu_info['cuda_device_count']:
                        logger.warning(f"PyTorch only detected {gpu_info['cuda_device_count']} GPUs but nvidia-smi shows {len(gpu_lines)}")
                        gpu_info['cuda_device_count'] = len(gpu_lines)
                        
                        # For p3dn.24xlarge, always set to 8 if we detect Tesla V100s
                        if any('Tesla V100' in line for line in gpu_lines) and len(gpu_lines) >= 4:
                            logger.info("Detected Tesla V100 GPUs, most likely on p3dn.24xlarge with 8 GPUs")
                            gpu_info['cuda_device_count'] = 8
                except Exception as e:
                    logger.warning(f"Failed to get GPU count from nvidia-smi: {e}")
            
            # On p3dn.24xlarge, add all 8 GPUs with minimal info to avoid errors
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
                
                # Skip detailed per-GPU detection which might fail
                return gpu_info
            
            # For all other cases, collect info about each CUDA device (safer approach)
            current_device = torch.cuda.current_device()
            for i in range(gpu_info['cuda_device_count']):
                try:
                    # Temporarily set device to avoid errors
                    torch.cuda.set_device(i)
                    device_name = torch.cuda.get_device_name(i)
                    
                    # Only get capabilities for current device to avoid errors
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
                    
                    # Try to get memory info only for current device
                    if i == current_device:
                        try:
                            free_mem, total_mem = torch.cuda.mem_get_info(i)
                            device_props['total_memory'] = total_mem
                            device_props['free_memory'] = free_mem
                            gpu_info['total_gpu_memory'] += total_mem
                        except:
                            # Older PyTorch versions or restricted environments
                            pass
                    
                    gpu_info['cuda_devices'].append(device_props)
                except Exception as e:
                    # If we can't get details for a specific GPU, add it with minimal info
                    logger.warning(f"Error getting detailed info for CUDA device {i}: {e}")
                    device_props = {
                        'index': i,
                        'name': f"GPU {i}",
                        'compute_capability': "unknown",
                    }
                    gpu_info['cuda_devices'].append(device_props)
                
            # Reset to original device
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
    # NCCL configuration for high-speed networks
    os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debug info
    os.environ["NCCL_IB_DISABLE"] = "0"  # Enable InfiniBand for interconnect
    os.environ["NCCL_IB_GID_INDEX"] = "3"  # Optimal for AWS Elastic Fabric Adapter (EFA)
    os.environ["NCCL_IB_HCA"] = "^mlx5_0"  # Use Mellanox adapters
    os.environ["NCCL_IB_TC"] = "106"  # Traffic class for IB
    os.environ["NCCL_IB_TIMEOUT"] = "23"  # Longer timeout for stability
    os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"  # Skip loopback, docker interfaces
    os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable GPU P2P
    os.environ["NCCL_P2P_LEVEL"] = "NVL"  # NVLink for P2P
    
    # CUDA optimizations
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match IDs with nvidia-smi
    
    # For conservative start, use only GPU 0 initially
    if conservative:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Start with just one GPU
        logger.info("Conservative mode: starting with only GPU 0")
    else:
        # Use all 8 GPUs right away
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(8))
        logger.info("Performance mode: using all 8 GPUs")
    
    os.environ["CUDA_CACHE_DISABLE"] = "0"  # Enable JIT cache
    os.environ["CUDA_AUTO_BOOST"] = "0"  # Disable autoboost for consistent performance
    
    # PyTorch specific optimizations
    os.environ["OMP_NUM_THREADS"] = "8"  # Limit OpenMP threads
    os.environ["KMP_BLOCKTIME"] = "0"  # Don't let OpenMP worker threads sleep
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"  # CPU affinity
    
    # Distributed training optimizations
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # Debug info for distributed
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"  # More verbose logging
    
    # Make sure we don't crash with multiple GPUs
    try:
        import torch
        torch.cuda.init()  # Explicitly initialize CUDA
        torch.cuda.set_device(0)  # Start with device 0
    except:
        logger.warning("Could not initialize CUDA directly")
    
    logger.info("Configured environment variables for AWS p3dn.24xlarge")


def setup_windows_gigabyte_environment(gigabyte_info):
    """
    Configure environment variables for optimal performance on Windows with Gigabyte GPUs.
    
    Args:
        gigabyte_info: Dictionary with Gigabyte GPU information
    """
    # CUDA optimizations for Windows
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match IDs with nvidia-smi
    os.environ["CUDA_CACHE_DISABLE"] = "0"  # Enable JIT cache
    
    # Windows-specific thread optimizations
    num_cpu_cores = os.cpu_count() or 4
    os.environ["OMP_NUM_THREADS"] = str(min(num_cpu_cores, 8))  # Limit OpenMP threads to avoid oversubscription
    os.environ["MKL_NUM_THREADS"] = str(min(num_cpu_cores, 8))  # Limit MKL threads similarly
    
    # Memory optimizations for Gigabyte GPUs
    # Allocate a larger portion of GPU memory upfront for better performance
    if gigabyte_info and gigabyte_info.get('vram'):
        vram_mb = gigabyte_info.get('vram')
        if vram_mb:
            # Set optimized parameters based on available VRAM
            if vram_mb >= 8000:  # 8GB or more
                # Higher memory budget for larger VRAM GPUs
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
            else:
                # More conservative for smaller VRAM GPUs
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # CUDNN optimizations
    os.environ["CUDNN_LOGINFO_DBG"] = "0"  # Disable verbose logging
    os.environ["CUDNN_LOGDEST_DBG"] = "stdout"  # Log to stdout for better diagnostics if needed
    
    # Windows-specific optimizations
    logger.info("Configured environment variables for Windows with Gigabyte GPU")
    gigabyte_info['optimized'] = True


def setup_apple_silicon_environment():
    """
    Configure environment variables for optimal performance on Apple Silicon (M1/M2/M3) hardware.
    """
    # MPS optimizations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
    
    # PyTorch performance optimizations for Apple Silicon
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # Use all cores for OpenMP
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())  # Use all cores for MKL
    
    # Memory management optimizations
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
    # Get device options from config or use auto-detection
    device_str = config["hyperparameters"].get("device", None)
    force_cuda = config.get("force_cuda", False)
    bypass_pytorch_cuda_check = config.get("bypass_pytorch_cuda_check", False)
    
    # Detect available GPU capabilities
    gpu_info = detect_gpu_capabilities()
    
    # Log detected capabilities
    logger.info(f"Detected system capabilities:")
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
    
    # Platform-specific optimizations - set up environment for specific platforms
    if gpu_info['is_aws'] and gpu_info['aws_instance_type'] == 'p3dn.24xlarge':
        # Configure for AWS p3dn.24xlarge with 8x Tesla V100 GPUs
        logger.info("Optimizing for AWS p3dn.24xlarge instance")
        setup_aws_p3dn_environment(conservative=True)  # Start with conservative mode for stability
        
        # Check if p3dn optimization is explicitly enabled in config
        aws_p3dn_optimization = config.get("hyperparameters", {}).get("aws_p3dn_optimization", True)
        
        # Get max GPUs to use if specified
        aws_max_gpus = config.get("hyperparameters", {}).get("aws_max_gpus", 8)
        if aws_max_gpus < 8:
            # Limit visible GPUs if user requested fewer
            import os
            visible_devices = ",".join(str(i) for i in range(aws_max_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            logger.info(f"Limiting to {aws_max_gpus} GPUs as specified in config: CUDA_VISIBLE_DEVICES={visible_devices}")
        
        # If we're on this specific instance type, always use CUDA
        device_str = "cuda"
        force_cuda = aws_p3dn_optimization  # Only force if optimization is enabled
        
    elif gpu_info['is_windows_gigabyte']:
        # Configure for Windows with Gigabyte GPUs
        logger.info(f"Optimizing for Windows with Gigabyte GPU: {gpu_info['gigabyte_gpu_info'].get('model', 'Unknown model')}")
        setup_windows_gigabyte_environment(gpu_info['gigabyte_gpu_info'])
        
        # Check if Windows Gigabyte optimization is explicitly enabled in config
        windows_gigabyte_optimization = config.get("hyperparameters", {}).get("windows_gigabyte_optimization", True)
        
        # Always use CUDA for Windows Gigabyte GPUs
        device_str = "cuda"
        force_cuda = windows_gigabyte_optimization
        
    elif gpu_info['apple_silicon']:
        # Configure for Apple Silicon
        logger.info(f"Optimizing for Apple Silicon: {gpu_info['chip_model'] or 'Unknown model'}")
        setup_apple_silicon_environment()
        
        # Default to MPS for Apple Silicon if not explicitly specified
        if not device_str:
            device_str = "mps"
    
    # If force_cuda is enabled but we can't detect CUDA normally
    if force_cuda and (not gpu_info['cuda_available'] or bypass_pytorch_cuda_check):
        logger.info("Force CUDA requested, applying advanced detection methods")
        
        # Advanced EC2 p3 instance checks - look for hardware signs
        # Try to get nvidia-smi output even if PyTorch doesn't see CUDA
        try:
            nvidia_smi = subprocess.check_output(['nvidia-smi', '-L'], stderr=subprocess.DEVNULL).decode('utf-8')
            if 'Tesla V100' in nvidia_smi:
                logger.info(f"Detected Tesla V100 GPUs from nvidia-smi:\n{nvidia_smi.strip()}")
                
                # Monkey patch torch.cuda to force availability if needed
                if bypass_pytorch_cuda_check:
                    logger.warning("MONKEY PATCHING torch.cuda for Tesla V100 detection - USE WITH CAUTION")
                    
                    # Save original functions
                    original_is_available = torch.cuda.is_available
                    original_device_count = torch.cuda.device_count
                    original_get_device_name = torch.cuda.get_device_name
                    
                    # Count V100 GPUs from nvidia-smi output
                    gpu_count = len(nvidia_smi.strip().split('\n'))
                    
                    # Save original functions
                    original_is_available = torch.cuda.is_available
                    original_device_count = torch.cuda.device_count
                    original_get_device_name = torch.cuda.get_device_name
                    
                    # Define override functions
                    def _is_available_override():
                        return True
                        
                    def _get_device_count_override():
                        return gpu_count
                        
                    def _get_device_name_override(device):
                        return "Tesla V100"
                    
                    # Apply patches
                    torch.cuda.is_available = _is_available_override
                    torch.cuda.device_count = _get_device_count_override
                    torch.cuda.get_device_name = _get_device_name_override
                    
                    # Update our cuda status after patching
                    gpu_info['cuda_available'] = True
                    gpu_info['cuda_device_count'] = gpu_count
                    
                    logger.info(f"After patching: CUDA available: {torch.cuda.is_available()}")
                    logger.info(f"After patching: CUDA device count: {torch.cuda.device_count()}")
                    
                    # Register cleanup function to restore original functions when the process exits
                    import atexit
                    def _restore_cuda_functions():
                        logger.info("Restoring original CUDA functions")
                        torch.cuda.is_available = original_is_available
                        torch.cuda.device_count = original_device_count
                        torch.cuda.get_device_name = original_get_device_name
                    
                    atexit.register(_restore_cuda_functions)
        except Exception as e:
            logger.warning(f"Advanced CUDA detection failed: {e}")
    
    # Final device selection with platform optimizations
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
    
    # Final device selection logic with precedence order
    if device_str == "cuda" and (gpu_info['cuda_available'] or force_cuda):
        # CUDA path for NVIDIA GPUs
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
        # MPS path for Apple Silicon
        device = torch.device("mps")
        device_info['type'] = 'mps'
        device_info['name'] = f"Apple {gpu_info['chip_model'] or 'Silicon'}"
        logger.info(f"Using MPS for Apple Silicon: {device_info['name']}")
        
    elif device_str == "cpu" or (not device_str and not gpu_info['cuda_available'] and not gpu_info['mps_available']):
        # CPU fallback
        device = torch.device("cpu")
        device_info['type'] = 'cpu'
        device_info['name'] = platform.processor() or "CPU"
        logger.info(f"Using CPU: {device_info['name']}")
        
    else:
        # Auto selection based on best available
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
    
    # Return both the device and detailed info about the configuration
    return device, device_info
