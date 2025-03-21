#!/usr/bin/env python
"""
Model training utilities for the ML experimentation framework.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loguru import logger
import random


def create_optimizer(model, config):
    """
    Create optimizer based on configuration.

    Args:
        model: PyTorch model
        config: Configuration dictionary

    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    device_type = model.embedding.weight.device.type
    optimizer_type = config["hyperparameters"].get("optimizer", "adam").lower()
    lr = float(config["hyperparameters"].get("lr", 1e-3))
    
    # Ensure weight_decay is a float to avoid type comparison errors
    weight_decay_raw = config["hyperparameters"].get("weight_decay", 0)
    weight_decay = float(weight_decay_raw) if weight_decay_raw is not None else 0.0

    # Device-specific optimizers
    if device_type == 'cuda':
        # For CUDA, use AdamW with higher weight decay
        if optimizer_type == "adamw" or optimizer_type == "adam":
            logger.info(f"Using AdamW optimizer with lr={lr} for CUDA")
            return optim.AdamW(model.parameters(), lr=lr, 
                              weight_decay=weight_decay if weight_decay > 0 else 1e-4,
                              eps=1e-8)
        elif optimizer_type == "sgd":
            # For SGD on CUDA, use Nesterov acceleration
            momentum = config["hyperparameters"].get("momentum", 0.9)
            logger.info(f"Using SGD with Nesterov, momentum={momentum}, lr={lr} for CUDA")
            return optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, 
                weight_decay=weight_decay, nesterov=True
            )
    elif device_type == 'mps':
        # For MPS, stick with Adam for stability
        if optimizer_type == "adam" or optimizer_type == "adamw":
            logger.info(f"Using Adam optimizer with lr={lr} for MPS")
            return optim.Adam(model.parameters(), lr=lr, 
                             weight_decay=weight_decay if weight_decay > 0 else 1e-5,
                             eps=1e-7)  # Slightly higher eps for stability
        elif optimizer_type == "sgd":
            # For SGD on MPS, use standard momentum
            momentum = config["hyperparameters"].get("momentum", 0.9)
            logger.info(f"Using SGD with momentum={momentum}, lr={lr} for MPS")
            return optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, 
                weight_decay=weight_decay, nesterov=False
            )
    
    # Generic fallbacks if device-specific options aren't matched
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        momentum = config["hyperparameters"].get("momentum", 0.9)
        return optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer {optimizer_type}, falling back to Adam")
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: Configured scheduler
    """
    use_scheduler = config["hyperparameters"].get("use_scheduler", False)

    if not use_scheduler:
        return None
        
    # Get device type from optimizer to customize scheduler
    device_type = 'cpu'
    for param_group in optimizer.param_groups:
        if len(param_group['params']) > 0:
            device_type = param_group['params'][0].device.type
            break
            
    scheduler_type = config["hyperparameters"].get("scheduler_type", "reduce_on_plateau")
    epochs = int(config["hyperparameters"].get("epochs", 100))
    
    # Device-specific scheduler configurations
    if device_type == 'cuda':
        # For CUDA, use more aggressive learning rate adjustments
        if scheduler_type == "reduce_on_plateau":
            patience = config["hyperparameters"].get("scheduler_patience", 5)
            factor = config["hyperparameters"].get("scheduler_factor", 0.2)  # More aggressive reduction
            logger.info(f"Using ReduceLROnPlateau for CUDA with patience={patience}, factor={factor}")
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience, verbose=True, 
                min_lr=1e-6, threshold=1e-4
            )
        elif scheduler_type == "cosine":
            logger.info(f"Using CosineAnnealingWarmRestarts for CUDA with T_0={epochs//3}")
            # Use warm restarts for CUDA
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=epochs//3, T_mult=1, eta_min=1e-6
            )
        elif scheduler_type == "step":
            step_size = int(config["hyperparameters"].get("scheduler_step_size", 10))
            factor = float(config["hyperparameters"].get("scheduler_factor", 0.3))
            logger.info(f"Using StepLR for CUDA with step_size={step_size}, gamma={factor}")
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=factor)
            
    elif device_type == 'mps':
        # For MPS, use gentler learning rate adjustments for stability
        if scheduler_type == "reduce_on_plateau":
            patience = config["hyperparameters"].get("scheduler_patience", 10)  # More patience
            factor = config["hyperparameters"].get("scheduler_factor", 0.5)  # Gentler reduction
            logger.info(f"Using ReduceLROnPlateau for MPS with patience={patience}, factor={factor}")
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience, verbose=True,
                min_lr=1e-6, threshold=1e-4
            )
        elif scheduler_type == "cosine":
            logger.info(f"Using CosineAnnealingLR for MPS with T_max={epochs}")
            # Standard cosine for MPS
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
        elif scheduler_type == "step":
            step_size = int(config["hyperparameters"].get("scheduler_step_size", 15))  # Larger step size
            factor = float(config["hyperparameters"].get("scheduler_factor", 0.7))  # Gentler reduction
            logger.info(f"Using StepLR for MPS with step_size={step_size}, gamma={factor}")
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=factor)
    
    # Generic fallbacks
    if scheduler_type == "reduce_on_plateau":
        patience = config["hyperparameters"].get("scheduler_patience", 5)
        factor = config["hyperparameters"].get("scheduler_factor", 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, verbose=True
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif scheduler_type == "step":
        step_size = int(config["hyperparameters"].get("scheduler_step_size", 10))
        factor = float(config["hyperparameters"].get("scheduler_factor", 0.5))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=factor)
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, not using scheduler")
        return None


def worker_init_fn(worker_id):
    """
    Initialize worker processes with a seed based on the worker id.
    This ensures reproducible data loading across worker processes.

    Args:
        worker_id: ID of the dataloader worker process
    """
    # Get base seed from PyTorch's initial seed (which comes from our global seed)
    worker_seed = torch.initial_seed() % 2**32

    # Each worker gets a different seed derived from the initial seed and worker_id
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def prepare_data_loaders(t_train, data_train, config, t_val=None, data_val=None, device=None, device_info=None):
    """
    Prepare data loaders for training and validation with reproducible behavior.
    Optimized for different hardware (CPU, CUDA, MPS) with support for DistributedDataParallel.
    Note: Data is kept on CPU so that pin_memory works. Move batches to GPU in the training loop.

    Args:
        t_train: Training time points
        data_train: Training data values
        config: Configuration dictionary
        t_val: Validation time points (optional)
        data_val: Validation data values (optional)
        device: PyTorch device (optional)
        device_info: Additional device information dictionary (optional)

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get seed from config for reproducible shuffling
    seed = config.get("random_seed", 42)

    # Get batch size with appropriate device-specific defaults
    if "batch_size" in config["hyperparameters"]:
        batch_size = config["hyperparameters"]["batch_size"]
    elif device is not None and device.type == 'cuda':
        # Larger batch size for CUDA
        batch_size = 256
    elif device is not None and device.type == 'mps':
        # Medium batch size for MPS
        batch_size = 128
    else:
        # Default for CPU
        batch_size = 64
    
    # Make sure batch size isn't larger than dataset
    batch_size = min(batch_size, len(t_train))
    
    # Get workers with appropriate defaults for each platform
    if "num_workers" in config["hyperparameters"]:
        num_workers = config["hyperparameters"]["num_workers"]
    elif device is not None and device.type == 'cuda':
        # For CUDA, use more workers (scaled by GPU count)
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_workers = min(4 * gpu_count, 16)  # Cap at reasonable value
    elif device is not None and device.type == 'mps':
        # For MPS, use moderate number of workers
        num_workers = 2
    else:
        # Default for CPU
        num_workers = 1
    
    # For AWS p3dn.24xlarge, optimize workers further
    if device_info and device_info.get('aws_instance') == 'p3dn.24xlarge':
        # Use 2 workers per GPU for optimal throughput
        num_workers = 16  # 2 workers × 8 GPUs
    
    # Convert to PyTorch tensors (keep them on CPU for pin_memory to work)
    x_tensor = torch.from_numpy(t_train).float().unsqueeze(-1)
    y_tensor = torch.from_numpy(data_train).float().unsqueeze(-1)

    # Create reproducible generator for shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Check if we should use distributed sampler
    use_distributed = (
        config["hyperparameters"].get("distributed_training", False) and 
        torch.cuda.is_available() and 
        torch.cuda.device_count() > 1 and
        torch.distributed.is_initialized()
    )
    
    train_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    
    if use_distributed:
        # For distributed training, use DistributedSampler
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
            seed=seed
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle when using sampler
            sampler=train_sampler,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            pin_memory=True,
            drop_last=config["hyperparameters"].get("drop_last", False),
            persistent_workers=num_workers > 0  # Keep workers alive between epochs
        )
        
        logger.info(f"Using DistributedSampler for training data - world size: {torch.distributed.get_world_size()}, rank: {torch.distributed.get_rank()}")
    else:
        # Standard DataLoader for single GPU or DataParallel
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            generator=g,
            drop_last=config["hyperparameters"].get("drop_last", False),
            pin_memory=True,
            persistent_workers=num_workers > 0  # Keep workers alive between epochs
        )
    
    logger.info(f"Created training DataLoader with batch_size={batch_size}, num_workers={num_workers}")

    val_loader = None
    if t_val is not None and data_val is not None:
        x_val_tensor = torch.from_numpy(t_val).float().unsqueeze(-1)
        y_val_tensor = torch.from_numpy(data_val).float().unsqueeze(-1)

        val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
        
        if use_distributed:
            # For distributed validation, use DistributedSampler with no shuffle
            from torch.utils.data.distributed import DistributedSampler
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False  # Don't shuffle validation
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn if num_workers > 0 else None,
                pin_memory=True,
                persistent_workers=num_workers > 0
            )
            
            logger.info("Using DistributedSampler for validation data")
        else:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,  # No need to shuffle validation data
                num_workers=num_workers,
                worker_init_fn=worker_init_fn if num_workers > 0 else None,
                pin_memory=True,
                persistent_workers=num_workers > 0
            )

    return train_loader, val_loader


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for regression.

    Args:
        y_true: Ground truth values as numpy array
        y_pred: Predicted values as numpy array

    Returns:
        dict: Dictionary of metrics
    """
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # R² score
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }
def train_model(model, t_train, data_train, config, device, validation_split=0.2):
    """
    Enhanced training loop with loss tracking, validation, and multi-GPU support.
    Optimized for both Apple Silicon and p3dn.24xlarge with 8x Tesla V100 GPUs.

    Args:
        model: A PyTorch model instance
        t_train: np.ndarray of shape (N,) with training 'x' values
        data_train: np.ndarray of shape (N,) with training 'y' values
        config: Dictionary of hyperparameters
        device: torch.device to run on
        validation_split: Fraction of data to use for validation

    Returns:
        dict: Training history (losses, metrics, etc.)
    """
    # Get device information if available
    device_info = None
    if isinstance(device, tuple) and len(device) == 2:
        # New format: device is actually a tuple of (device, device_info)
        device, device_info = device
    
    # Check for multi-GPU capabilities
    use_multi_gpu = config["hyperparameters"].get("multigpu", False) and torch.cuda.device_count() > 1
    
    # If we have device_info, use it to get more accurate GPU count
    if device_info and device_info.get('is_multi_gpu', False):
        num_gpus = device_info.get('gpu_count', torch.cuda.device_count())
    else:
        num_gpus = torch.cuda.device_count() if use_multi_gpu else 1
    
    # Set up gradient accumulation
    gradient_accumulation_steps = int(config["hyperparameters"].get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps > 1:
        logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    
    # Detect if we're running on AWS p3dn.24xlarge
    is_p3dn = False
    if device_info and device_info.get('aws_instance') == 'p3dn.24xlarge':
        logger.info("Detected p3dn.24xlarge instance - optimizing for 8x Tesla V100 GPUs")
        is_p3dn = True
        # Force settings for p3dn.24xlarge
        use_multi_gpu = True
        num_gpus = 8
    
    # Record if we're using DDP for specific optimizations later
    using_ddp = False
    
    if use_multi_gpu and num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs for training")
        
        # Handle 'auto' option for distributed_training
        distributed_training = config["hyperparameters"].get("distributed_training", False)
        if distributed_training == "auto":
            # Auto-detect: use distributed for p3dn.24xlarge, DataParallel otherwise
            try_distributed = is_p3dn or (device_info and device_info.get('aws_instance') is not None)
            logger.info(f"Auto-detected distributed_training={try_distributed}")
        else:
            # Use explicit setting
            try_distributed = distributed_training
        
        # For AWS p3dn.24xlarge, we'll try harder to use DDP for better performance
        if is_p3dn:
            logger.info("AWS p3dn.24xlarge detected - optimizing for 8x Tesla V100 GPUs")
            try_distributed = True
        
        # First try DistributedDataParallel for better performance
        if try_distributed:
            try:
                # Try to initialize distributed backend if not already done
                if not torch.distributed.is_initialized():
                    # For AWS p3dn.24xlarge, use NCCL backend with specific settings
                    if is_p3dn:
                        import os
                        # Set environment variables specifically for NCCL on p3dn
                        os.environ["NCCL_DEBUG"] = "INFO"
                        os.environ["NCCL_IB_DISABLE"] = "0"
                        os.environ["NCCL_IB_GID_INDEX"] = "3"
                        os.environ["NCCL_IB_TIMEOUT"] = "23"
                        os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"
                        
                        # Initialize with file method which doesn't need env vars
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        file_path = os.path.join(temp_dir, "torch_distributed_init")
                        init_method = f"file://{file_path}"
                        
                        # Single-node multi-GPU setup parameters
                        world_size = num_gpus
                        rank = 0  # Master process
                        
                        logger.info(f"Initializing distributed: world_size={world_size}, rank={rank}, init_method={init_method}")
                        try:
                            torch.distributed.init_process_group(
                                backend="nccl",
                                init_method=init_method,
                                world_size=world_size,
                                rank=rank
                            )
                            logger.info("Successfully initialized distributed process group with file:// method")
                        except Exception as e1:
                            logger.warning(f"File-based initialization failed: {e1}, trying TCP")
                            
                            # Try TCP method as fallback
                            try:
                                # Try with tcp init method on localhost
                                init_method = "tcp://127.0.0.1:29500"
                                logger.info(f"Trying TCP initialization: world_size={world_size}, rank={rank}")
                                torch.distributed.init_process_group(
                                    backend="nccl",
                                    init_method=init_method,
                                    world_size=world_size,
                                    rank=rank
                                )
                                logger.info("Successfully initialized distributed process group with TCP method")
                            except Exception as e2:
                                logger.warning(f"TCP initialization failed: {e2}")
                                logger.warning("All distributed initialization methods failed, falling back to DataParallel")
                                raise RuntimeError("Could not initialize process group with any method")
                    else:
                        # For non-p3dn instances, try a simpler approach
                        try:
                            # Use TCP method with localhost
                            init_method = "tcp://127.0.0.1:29500"
                            world_size = num_gpus
                            rank = 0
                            
                            logger.info(f"Initializing distributed: world_size={world_size}, rank={rank}, init_method={init_method}")
                            torch.distributed.init_process_group(
                                backend="nccl",
                                init_method=init_method,
                                world_size=world_size,
                                rank=rank
                            )
                            logger.info("Successfully initialized distributed process group")
                        except Exception as e:
                            logger.warning(f"Distributed initialization failed: {e}")
                            raise RuntimeError(f"Could not initialize distributed process group: {e}")
                
                # If we get here, initialization succeeded
                local_rank = torch.distributed.get_rank()
                torch.cuda.set_device(local_rank)
                logger.info(f"Setting CUDA device to local rank: {local_rank}")
                
                # Use DDP for most efficient multi-GPU training
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = DDP(model, device_ids=[local_rank])
                logger.info("Using DistributedDataParallel for multi-GPU training")
                using_ddp = True
                
            except (ImportError, RuntimeError) as e:
                # Fall back to DataParallel if distributed fails
                logger.warning(f"Failed to initialize distributed training: {e}")
                logger.warning("Falling back to DataParallel (less efficient but more compatible)")
                try:
                    from torch.nn.parallel import DataParallel
                    model = DataParallel(model)
                    logger.info("Using DataParallel for multi-GPU training")
                except Exception as dp_error:
                    logger.error(f"Failed to initialize DataParallel: {dp_error}")
                    logger.warning("Will continue with single GPU training")
        else:
            # Use DataParallel for simpler multi-GPU training
            try:
                from torch.nn.parallel import DataParallel
                model = DataParallel(model)
                logger.info("Using DataParallel for multi-GPU training")
            except Exception as dp_error:
                logger.error(f"Failed to initialize DataParallel: {dp_error}")
                logger.warning("Will continue with single GPU training")
    # Get the random seed for reproducible validation split
    seed = config.get("random_seed", 42)

    # Split data into training and validation if needed
    if validation_split > 0:
        # Save the current random state and set the seed for reproducibility
        rng_state = np.random.get_state()
        np.random.seed(seed)

        val_size = int(len(t_train) * validation_split)
        indices = np.random.permutation(len(t_train))
        val_indices, train_indices = indices[:val_size], indices[val_size:]

        t_val, data_val = t_train[val_indices], data_train[val_indices]
        t_train, data_train = t_train[train_indices], data_train[train_indices]
        has_validation = True

        # Restore the random state
        np.random.set_state(rng_state)
    else:
        t_val, data_val = None, None
        has_validation = False

    # Prepare data loaders with device-specific optimizations
    config["hyperparameters"]["pin_memory"] = True  # Use pin_memory by default
    
    # Device-specific optimizations
    if device and device.type == 'cuda':
        # CUDA-specific settings
        logger.info("Using CUDA-optimized parameters")
        # Larger batch size for CUDA
        if "batch_size" not in config["hyperparameters"]:
            config["hyperparameters"]["batch_size"] = 128
        # More workers for CUDA (1 per GPU + 2)
        config["hyperparameters"]["num_workers"] = min(torch.cuda.device_count() * 2 + 2, 8)
        # Higher learning rate for CUDA
        if "lr" not in config["hyperparameters"]:
            config["hyperparameters"]["lr"] = 1e-3
        # Clip at 5.0 for stability but allow larger gradients
        config["hyperparameters"]["clip_value"] = 5.0
        # Enable AMP for faster training
        config["hyperparameters"]["use_amp"] = True
        
    elif device and device.type == 'mps':
        # MPS-specific settings (Apple Silicon)
        logger.info("Using MPS-optimized parameters")
        # Moderate batch size for MPS
        if "batch_size" not in config["hyperparameters"]:
            config["hyperparameters"]["batch_size"] = 64
        # 2 workers is optimal for most Apple chips
        config["hyperparameters"]["num_workers"] = 2
        # Slightly lower learning rate for MPS stability
        if "lr" not in config["hyperparameters"]:
            config["hyperparameters"]["lr"] = 5e-4
        # Tighter gradient clipping for MPS
        config["hyperparameters"]["clip_value"] = 1.0
        # Try to enable AMP for newer PyTorch versions
        config["hyperparameters"]["use_amp"] = True
        
    else:
        # CPU-specific settings
        logger.info("Using CPU-optimized parameters")
        # Smaller batch size for CPU
        if "batch_size" not in config["hyperparameters"]:
            config["hyperparameters"]["batch_size"] = 32
        # Just 1 worker for CPU to avoid overhead
        config["hyperparameters"]["num_workers"] = 1
        # Lower learning rate for CPU
        if "lr" not in config["hyperparameters"]:
            config["hyperparameters"]["lr"] = 1e-4
        # Standard clipping for CPU
        config["hyperparameters"]["clip_value"] = 1.0
    
    # Log the device-specific settings
    logger.info(f"Batch size: {config['hyperparameters']['batch_size']}")
    logger.info(f"Workers: {config['hyperparameters']['num_workers']}")
    logger.info(f"Learning rate: {config['hyperparameters']['lr']}")
    logger.info(f"Gradient clip: {config['hyperparameters']['clip_value']}")
    
    train_loader, val_loader = prepare_data_loaders(t_train, data_train, config, t_val, data_val, device, device_info)

    # Set up loss function, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    num_epochs = int(config["hyperparameters"].get("epochs", 10))
    
    # Enable Automatic Mixed Precision (AMP) for faster training
    # Make sure we parse the use_amp setting correctly from the config
    use_amp_raw = config["hyperparameters"].get("use_amp", False)
    use_amp = (use_amp_raw is True or 
              (isinstance(use_amp_raw, str) and use_amp_raw.lower() == 'true'))
    
    # Create gradient scaler for mixed precision if needed
    if device.type == 'cuda' and use_amp:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("AMP enabled for CUDA device")
    else:
        use_amp = False  # Disable AMP for non-CUDA devices
        scaler = None
        logger.info(f"AMP disabled for device type: {device.type}")

    # History tracking
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [] if has_validation else None,
        "learning_rate": [],
        "metrics": [] if has_validation else None,
        "epoch_times": [],
    }

    # Early stopping parameters
    early_stopping = config["hyperparameters"].get("early_stopping", False)
    if early_stopping:
        best_loss = float("inf")
        patience = config["hyperparameters"].get("early_stopping_patience", 10)
        min_delta = config["hyperparameters"].get("early_stopping_min_delta", 0.001)
        counter = 0

    # Single epoch loop
    for epoch in range(num_epochs):
        start_time = time.time()
        history["epochs"].append(epoch)

        # Training phase
        model.train()
        running_loss = 0.0
        
        # Track information for gradient accumulation
        batch_count = 0
        accumulated_loss = 0
        accumulated_samples = 0
        
        for x_batch, y_batch in train_loader:
            batch_count += 1
            
            # Move training batch to device with non_blocking for parallel transfer
            if device is not None:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
            # Zero gradients only at the start of accumulation cycle
            if (batch_count - 1) % gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with optimizations for CUDA and MPS
            if device.type == 'cuda' and use_amp:
                try:
                    # Try the newer API first
                    with torch.amp.autocast(device_type='cuda'):
                        preds = model(x_batch)
                        # Scale loss by accumulation steps for consistent gradient values
                        loss = criterion(preds, y_batch) / gradient_accumulation_steps
                except TypeError:
                    # Fall back to the older API if device_type param not supported
                    with torch.amp.autocast():
                        preds = model(x_batch)
                        loss = criterion(preds, y_batch) / gradient_accumulation_steps
            elif device.type == 'mps' and use_amp:
                # Try MPS AMP if enabled
                try:
                    # First try newer API with device_type
                    with torch.amp.autocast(device_type='mps'):
                        preds = model(x_batch)
                        loss = criterion(preds, y_batch) / gradient_accumulation_steps
                except (ValueError, RuntimeError, AttributeError, TypeError):
                    # Fallback to standard computation
                    preds = model(x_batch)
                    loss = criterion(preds, y_batch) / gradient_accumulation_steps
            else:
                # CPU path or MPS without AMP
                preds = model(x_batch)
                
                # Check for NaN in preds
                if torch.isnan(preds).any():
                    logger.warning("NaN detected in model outputs, replacing with zeros")
                    preds = torch.where(torch.isnan(preds), torch.zeros_like(preds), preds)
                    
                loss = criterion(preds, y_batch) / gradient_accumulation_steps
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning("NaN loss detected, using zero loss instead")
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
            
            # Track loss for reporting (use the unscaled value)
            batch_loss = loss.item() * gradient_accumulation_steps
            running_loss += batch_loss * x_batch.size(0)
            accumulated_loss += batch_loss * x_batch.size(0)
            accumulated_samples += x_batch.size(0)
            
            # Backward pass with gradient scaling for mixed precision
            if (device.type == 'cuda' or device.type == 'mps') and use_amp and scaler is not None:
                # GPU path with mixed precision
                scaler.scale(loss).backward()
                
                # Only perform optimization step at the end of accumulation cycle
                if (batch_count % gradient_accumulation_steps == 0) or (batch_count == len(train_loader)):
                    # Optional gradient clipping
                    if config["hyperparameters"].get("clip_gradients", True):
                        clip_value = float(config["hyperparameters"].get("clip_value", 1.0))
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    
                    # Update weights with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Log accumulated loss for better monitoring with multi-GPU
                    if num_gpus > 1 and gradient_accumulation_steps > 1:
                        avg_loss = accumulated_loss / accumulated_samples if accumulated_samples > 0 else 0
                        logger.info(f"Accumulated step {batch_count//gradient_accumulation_steps}, " 
                                   f"batch loss: {avg_loss:.6f}, samples: {accumulated_samples}")
                        accumulated_loss = 0
                        accumulated_samples = 0
            else:
                # Standard backward pass (CPU or MPS without AMP)
                loss.backward()
                
                # Only perform optimization step at the end of accumulation cycle
                if (batch_count % gradient_accumulation_steps == 0) or (batch_count == len(train_loader)):
                    # Apply standard gradient clipping for all devices
                    use_gradient_clipping = config["hyperparameters"].get("clip_gradients", True)
                    if use_gradient_clipping:
                        clip_value = float(config["hyperparameters"].get("clip_value", 1.0))
                        
                        # Debug check for NaN gradients with improved tracing
                        has_nan_grad = False
                        for name, param in model.named_parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                has_nan_grad = True
                                logger.warning(f"NaN/Inf gradient detected in {name} before clipping")
                                if device.type == 'cuda':
                                    # Check CUDA memory status when NaN detected on CUDA device
                                    try:
                                        free_mem, total_mem = torch.cuda.mem_get_info()
                                        logger.warning(f"CUDA memory: {free_mem/1e9:.2f}GB free / {total_mem/1e9:.2f}GB total")
                                    except:
                                        logger.warning("Could not retrieve CUDA memory info")
                        
                        # Apply gradient clipping to all params
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    
                    # Standard weight update
                    optimizer.step()
                    
                    # Synchronize GPUs if using multiple GPU training
                    if device.type == 'cuda' and num_gpus > 1:
                        torch.cuda.synchronize()
                    
                    # Check for NaN in parameters after update
                    has_nan = False
                    for param in model.parameters():
                        if torch.isnan(param.data).any():
                            has_nan = True
                            logger.warning("NaN detected in model parameters after update, replacing with zeros")
                            param.data = torch.where(torch.isnan(param.data), torch.zeros_like(param.data), param.data)
                    
                    # Log accumulated loss for better monitoring with multi-GPU
                    if num_gpus > 1 and gradient_accumulation_steps > 1:
                        avg_loss = accumulated_loss / accumulated_samples if accumulated_samples > 0 else 0
                        logger.info(f"Accumulated step {batch_count//gradient_accumulation_steps}, " 
                                   f"batch loss: {avg_loss:.6f}, samples: {accumulated_samples}")
                        accumulated_loss = 0
                        accumulated_samples = 0
        epoch_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # Validation phase
        if has_validation and val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    # Move validation batch to device with non_blocking for parallel transfer
                    if device is not None:
                        x_val = x_val.to(device, non_blocking=True)
                        y_val = y_val.to(device, non_blocking=True)
                    
                    # Mixed precision for validation too
                    if device.type == 'cuda' and use_amp:
                        try:
                            # Try the newer API first
                            with torch.amp.autocast(device_type='cuda'):
                                val_preds = model(x_val)
                                val_batch_loss = criterion(val_preds, y_val).item()
                        except TypeError:
                            # Fall back to the older API if device_type param not supported
                            with torch.amp.autocast():
                                val_preds = model(x_val)
                                val_batch_loss = criterion(val_preds, y_val).item()
                    elif device.type == 'mps' and use_amp:
                        # Try MPS AMP for validation if enabled
                        try:
                            # First try newer API with device_type
                            with torch.amp.autocast(device_type='mps'):
                                val_preds = model(x_val)
                                val_batch_loss = criterion(val_preds, y_val).item()
                        except (ValueError, RuntimeError, AttributeError, TypeError):
                            # Fallback if autocast fails
                            val_preds = model(x_val)
                            val_loss_tensor = criterion(val_preds, y_val)
                            val_batch_loss = val_loss_tensor.item()
                    else:
                        # CPU path or MPS without AMP
                        val_preds = model(x_val)
                        val_loss_tensor = criterion(val_preds, y_val)
                        val_batch_loss = val_loss_tensor.item()
                    
                    val_loss += val_batch_loss * x_val.size(0)
                    
                    # Move predictions and targets to CPU for metric computation
                    # Use .detach() for more efficient memory management
                    all_preds.append(val_preds.detach().cpu().numpy())
                    all_targets.append(y_val.detach().cpu().numpy())
            val_loss = val_loss / len(val_loader.dataset)
            history["val_loss"].append(val_loss)

            all_preds = np.vstack(all_preds).flatten()
            all_targets = np.vstack(all_targets).flatten()
            metrics = compute_metrics(all_targets, all_preds)
            history["metrics"].append(metrics)

            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}"
            )

            # Scheduler step for ReduceLROnPlateau
            if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

            # Early stopping check
            if early_stopping:
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
        else:
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.6f}"
            )
            # Scheduler step for non-plateau schedulers
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        epoch_time = time.time() - start_time
        history["epoch_times"].append(epoch_time)

    return history