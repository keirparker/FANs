#!/bin/bash
# Script to fix CUDA detection on p3.8x EC2 instance
# Run this script before starting your PyTorch application

echo "=== CUDA Detection Fix Script for p3.8x EC2 ==="

# 1. Set environment variables
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Set environment variables:"
echo "CUDA_HOME=$CUDA_HOME"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"

# 2. Check if nvidia-smi works
echo -e "\nChecking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "NVIDIA GPU detected."
else
    echo "nvidia-smi not found. Installing NVIDIA driver..."
    # Try to install NVIDIA driver (AWS Deep Learning AMI might need different commands)
    sudo apt-get update -y &> /dev/null || sudo yum update -y &> /dev/null
    sudo apt-get install -y nvidia-driver-latest || sudo yum install -y nvidia-driver-latest
    echo "Please reboot the instance after installation."
fi

# 3. Check CUDA installation
echo -e "\nChecking CUDA installation..."
if [ -d "$CUDA_HOME" ]; then
    echo "CUDA found at $CUDA_HOME"
    ls -la $CUDA_HOME
else
    echo "CUDA not found at $CUDA_HOME"
    
    # Try to find CUDA
    CUDA_PATHS=$(find /usr -name "cuda*" -type d -maxdepth 2 2>/dev/null)
    if [ -n "$CUDA_PATHS" ]; then
        echo "Found possible CUDA installations:"
        echo "$CUDA_PATHS"
        
        # Set to the first found CUDA path
        FIRST_CUDA=$(echo "$CUDA_PATHS" | head -1)
        export CUDA_HOME=$FIRST_CUDA
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        echo "Set CUDA_HOME to $CUDA_HOME"
    else
        echo "No CUDA installation found."
    fi
fi

# 4. Check PyTorch CUDA compatibility
echo -e "\nChecking PyTorch CUDA compatibility..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f'GPU 0 name: {torch.cuda.get_device_name(0)}')
else:
    print('Cannot get GPU name, no CUDA device available')
print(f'PyTorch path: {torch.__file__}')
"

# 5. Install correct PyTorch version if needed
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo -e "\nCUDA not available in PyTorch. Installing compatible version..."
    
    # Detect CUDA version
    if [ -d "$CUDA_HOME" ]; then
        CUDA_VERSION=$(cat $CUDA_HOME/version.txt 2>/dev/null | grep -o "[0-9]*\.[0-9]*" | head -1)
        if [ -z "$CUDA_VERSION" ]; then
            # Try another method
            CUDA_VERSION=$($CUDA_HOME/bin/nvcc --version 2>/dev/null | grep -o "release [0-9]*\.[0-9]*" | awk '{print $2}')
        fi
        
        echo "Detected CUDA version: $CUDA_VERSION"
        CUDA_SHORT_VERSION=$(echo $CUDA_VERSION | sed 's/\.//')
        
        # Install appropriate PyTorch version
        if [ "$CUDA_SHORT_VERSION" = "124" ]; then
            pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
        elif [ "$CUDA_SHORT_VERSION" = "118" ] || [ "$CUDA_SHORT_VERSION" = "117" ]; then
            pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
        elif [ "$CUDA_SHORT_VERSION" = "116" ] || [ "$CUDA_SHORT_VERSION" = "115" ]; then
            pip install torch==2.0.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
        else
            echo "Unknown CUDA version. Trying with latest PyTorch..."
            pip install torch torchvision torchaudio
        fi
    else
        echo "Cannot determine CUDA version. Installing latest PyTorch..."
        pip install torch torchvision torchaudio
    fi
    
    # Verify after installation
    echo -e "\nVerifying PyTorch installation..."
    python -c "
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda}')
    "
fi

echo -e "\n=== CUDA detection script completed ==="
echo "If PyTorch still can't detect CUDA:"
echo "1. Your code has a fallback mechanism with force_cuda: true in config.yml"
echo "2. You may need to reboot the instance after driver installation"
echo "3. Try running nvidia-docker if available on your instance"
echo

# Make the script add itself to .bashrc for automatic execution on login
if ! grep -q "fix_cuda.sh" ~/.bashrc; then
    echo -e "\nAdding this script to .bashrc for automatic execution"
    echo "# CUDA setup" >> ~/.bashrc
    echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
    echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID" >> ~/.bashrc
fi

# Return true so script doesn't fail if used with source command
true