#!/usr/bin/env python
"""
Helper script for running experiments on EC2 p3dn.24xlarge instances.
This ensures all CUDA environments are properly set up before the main script executes.
"""

import os
import sys
import subprocess
import time

# =====================================================================
# CONFIGURATION OPTIONS - Modify these settings as needed
# =====================================================================

# GPU Configuration
USE_ALL_GPUS = True         # Try to use all available GPUs
FALLBACK_MODE = True        # If True, auto-fallback to fewer GPUs if errors occur
MIN_GPUS = 1                # Minimum GPUs to use if fallbacks happen
MAX_GPUS = 8                # Maximum GPUs to use (up to 8)

# Stability Options - Addressing NaN Loss
SAFE_MODE = False           # When True, uses DataParallel instead of DDP (more stable)
USE_SINGLE_GPU = False      # Force single GPU even if more are available
GRADIENT_ACCUM_STEPS = 4    # Accumulate gradients for more stable optimization
LEARNING_RATE = 5e-5        # Lower learning rate for stability (vs default 1e-4)
CLIP_VALUE = 0.5            # Gradient clip value (lower value = more stability)

# =====================================================================
# END OF CONFIGURATION
# =====================================================================

# Set environment variables for CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# We'll determine CUDA_VISIBLE_DEVICES based on available GPUs
# Initially, we don't set it so we can detect all GPUs

# Set NCCL environment variables for better performance
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "0"  # Enable InfiniBand
os.environ["NCCL_IB_GID_INDEX"] = "3"
os.environ["NCCL_IB_TIMEOUT"] = "23"
os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"
os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable GPU P2P
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # More verbose distributed logging

# Also force initial config to be safer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # Avoid fragmentation

print(f"Setting up EC2 environment for PyTorch CUDA execution")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER', 'not set')}")

# Check for CUDA availability
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print("\nCUDA Device Information:")
    print(result.stdout)
except Exception as e:
    print(f"Error getting CUDA device information: {e}")

# Initialize and test CUDA - write a tiny test script
test_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    # Try to create a small tensor on GPU
    try:
        x = torch.ones(10, device='cuda:0')
        print("Successfully created tensor on CUDA device")
    except Exception as e:
        print(f"Error creating tensor on CUDA: {e}")
"""

with open("cuda_test.py", "w") as f:
    f.write(test_script)

# Run the test script
print("\nTesting CUDA initialization:")
try:
    subprocess.run([sys.executable, "cuda_test.py"], check=True)
except Exception as e:
    print(f"Error running CUDA test: {e}")

# First, detect available GPUs
try:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    gpu_lines = result.stdout.strip().split('\n')
    available_gpus = len(gpu_lines)
    print(f"\nDetected {available_gpus} available GPU(s):")
    for i, line in enumerate(gpu_lines):
        print(f"  GPU {i}: {line.strip()}")
except Exception as e:
    print(f"Error detecting GPUs: {e}")
    available_gpus = 1  # Assume at least 1 GPU

# Determine how many GPUs to use based on configuration
if not USE_ALL_GPUS:
    # Use just one GPU if multi-GPU is disabled
    gpus_to_use = 1
    print("Multi-GPU mode disabled. Using 1 GPU only.")
else:
    # Use all available GPUs up to MAX_GPUS
    gpus_to_use = min(available_gpus, MAX_GPUS)
    print(f"Using {gpus_to_use} out of {available_gpus} available GPUs (max configured: {MAX_GPUS})")

# Check if the CUDA test script succeeds with all GPUs
test_success = True
if gpus_to_use > 1:
    # First, try with all GPUs
    visible_devices = ",".join(str(i) for i in range(gpus_to_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    print(f"Testing with CUDA_VISIBLE_DEVICES={visible_devices}")
    
    # Create a more aggressive test script for multi-GPU configuration
    multi_gpu_test = """
import torch
import os
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    
    # Try to create a tensor on each GPU
    try:
        # Initialize CUDA explicitly
        torch.cuda.init()
        
        # Test each GPU
        for i in range(torch.cuda.device_count()):
            print(f"Testing GPU {i}...")
            torch.cuda.set_device(i)
            x = torch.ones(10, device=f'cuda:{i}')
            print(f"Successfully created tensor on CUDA device {i}")
            
            # Try GPU-to-GPU transfer for multi-GPU
            if i > 0:
                y = x.to(f'cuda:0')
                z = y.to(f'cuda:{i}')
                print(f"Successfully tested device transfer between GPU 0 and GPU {i}")
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)  # Indicate failure
"""
    
    with open("cuda_multi_gpu_test.py", "w") as f:
        f.write(multi_gpu_test)
    
    print("\nTesting multi-GPU CUDA initialization:")
    try:
        result = subprocess.run([sys.executable, "cuda_multi_gpu_test.py"], 
                                capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            test_success = False
            print("Multi-GPU test failed!")
            print(result.stderr)
        else:
            print("Multi-GPU test successful!")
    except Exception as e:
        print(f"Error running multi-GPU CUDA test: {e}")
        test_success = False
    
    # Clean up test script
    try:
        os.remove("cuda_multi_gpu_test.py")
    except:
        pass

# If multi-GPU test failed and fallback mode is enabled, reduce GPUs
if not test_success and FALLBACK_MODE and gpus_to_use > MIN_GPUS:
    print(f"\nFalling back to fewer GPUs due to test failure")
    
    # Try with half the GPUs first, then MIN_GPUS
    gpus_to_use = max(MIN_GPUS, gpus_to_use // 2)
    visible_devices = ",".join(str(i) for i in range(gpus_to_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    print(f"Now using {gpus_to_use} GPUs: CUDA_VISIBLE_DEVICES={visible_devices}")
else:
    # Keep GPU configuration as is
    visible_devices = ",".join(str(i) for i in range(gpus_to_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    print(f"Setting CUDA_VISIBLE_DEVICES={visible_devices} for {gpus_to_use} GPUs")

# Update the config file to match our GPU settings
try:
    with open("configs/config.yml", "r") as f:
        config_content = f.read()
    
    # Update aws_max_gpus setting
    if "aws_max_gpus:" in config_content:
        config_content = config_content.replace(
            "aws_max_gpus: 8", 
            f"aws_max_gpus: {gpus_to_use}"
        )
        
    # Also make sure distributed_training is set appropriately
    if gpus_to_use > 1:
        # For multi-GPU, set distributed_training to auto or true
        if "distributed_training:" in config_content:
            config_content = config_content.replace(
                "distributed_training: false", 
                "distributed_training: auto"
            )
    else:
        # For single GPU, set to false
        if "distributed_training:" in config_content:
            config_content = config_content.replace(
                "distributed_training: auto", 
                "distributed_training: false"
            )
    
    with open("configs/config.yml", "w") as f:
        f.write(config_content)
    
    print(f"Updated config.yml to use {gpus_to_use} GPUs")
except Exception as e:
    print(f"Error updating config file: {e}")

# Clean up test file
try:
    os.remove("cuda_test.py")
except:
    pass

# Run the actual script
print("\nRunning main script...")
time.sleep(1)  # Give time for things to stabilize

# Pass any additional command line arguments to runner.py
cmd_args = [sys.executable, "runner.py"] + sys.argv[1:]
print(f"Executing: {' '.join(cmd_args)}")

# Additional environment variables to control training
if USE_SINGLE_GPU:
    # Set distributed_training to false when using single GPU
    os.environ["DISABLE_DISTRIBUTED"] = "1"
elif SAFE_MODE:
    # When in safe mode with multi-GPU, use DataParallel instead of DDP
    os.environ["USE_DATAPARALLEL"] = "1"

# NaN loss prevention options
os.environ["GRADIENT_ACCUMULATION_STEPS"] = str(GRADIENT_ACCUM_STEPS)
os.environ["OVERRIDE_LEARNING_RATE"] = str(LEARNING_RATE)
os.environ["OVERRIDE_CLIP_VALUE"] = str(CLIP_VALUE)
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # More verbose distributed logging
os.environ["NAN_DETECTION"] = "1"  # Enable NaN detection and correction

# Execute the runner script with the specified arguments
os.execv(sys.executable, cmd_args)