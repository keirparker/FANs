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


def create_optimizer(model, config):
    """
    Create optimizer based on configuration.

    Args:
        model: PyTorch model
        config: Configuration dictionary

    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    optimizer_type = config["hyperparameters"].get("optimizer", "adam").lower()
    lr = config["hyperparameters"].get("lr", 1e-3)
    weight_decay = config["hyperparameters"].get("weight_decay", 0)

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

    scheduler_type = config["hyperparameters"].get(
        "scheduler_type", "reduce_on_plateau"
    )
    patience = config["hyperparameters"].get("scheduler_patience", 5)
    factor = config["hyperparameters"].get("scheduler_factor", 0.5)

    if scheduler_type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, verbose=True
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["hyperparameters"].get("epochs", 10)
        )
    elif scheduler_type == "step":
        step_size = config["hyperparameters"].get("scheduler_step_size", 10)
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


def prepare_data_loaders(
    t_train, data_train, config, t_val=None, data_val=None, device=None
):
    """
    Prepare data loaders for training and validation with reproducible behavior.

    Args:
        t_train: Training time points
        data_train: Training data values
        config: Configuration dictionary
        t_val: Validation time points (optional)
        data_val: Validation data values (optional)
        device: PyTorch device (optional)

    Returns:
        tuple: (train_loader, val_loader)
    """
    import random
    
    # Get seed from config for reproducible shuffling
    seed = config.get("random_seed", 42)
    
    batch_size = config["hyperparameters"].get("batch_size", 64)
    batch_size = min(batch_size, len(t_train))
    num_workers = config["hyperparameters"].get("num_workers", 0)

    # Convert to PyTorch tensors
    x_tensor = torch.from_numpy(t_train).float().unsqueeze(-1)
    y_tensor = torch.from_numpy(data_train).float().unsqueeze(-1)

    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

    # Create reproducible generator for shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
        drop_last=False,  # More deterministic to keep all samples
        pin_memory=True if device is not None and device.type == 'cuda' else False
    )

    val_loader = None
    if t_val is not None and data_val is not None:
        x_val_tensor = torch.from_numpy(t_val).float().unsqueeze(-1)
        y_val_tensor = torch.from_numpy(data_val).float().unsqueeze(-1)

        if device is not None:
            x_val_tensor = x_val_tensor.to(device)
            y_val_tensor = y_val_tensor.to(device)

        val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            pin_memory=True if device is not None and device.type == 'cuda' else False
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

    # Mean Absolute Percentage Error
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }

def train_model(model, t_train, data_train, config, device, validation_split=0.2):
    """
    Enhanced training loop with loss tracking and validation.

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
    # Get the random seed for reproducible validation split
    seed = config.get("random_seed", 42)
    
    # Split data into training and validation if needed
    if validation_split > 0:
        # Set a specific seed state for validation split to ensure reproducibility
        # Save the current random state
        rng_state = np.random.get_state()
        
        # Set seed for this operation
        np.random.seed(seed)
        
        # Generate reproducible permutation
        val_size = int(len(t_train) * validation_split)
        indices = np.random.permutation(len(t_train))
        val_indices, train_indices = indices[:val_size], indices[val_size:]

        t_val, data_val = t_train[val_indices], data_train[val_indices]
        t_train, data_train = t_train[train_indices], data_train[train_indices]
        has_validation = True
        
        # Restore the random state to not affect other operations
        np.random.set_state(rng_state)
    else:
        t_val, data_val = None, None
        has_validation = False

    # Prepare data loaders - disable pin_memory on CUDA since tensors are already on device
    pin_memory = False if device and device.type == 'cuda' else True
    config["hyperparameters"]["pin_memory"] = pin_memory
    
    train_loader, val_loader = prepare_data_loaders(
        t_train, data_train, config, t_val, data_val, device
    )

    # Set up loss function
    criterion = nn.MSELoss()

    # Set up optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Training parameters
    num_epochs = config["hyperparameters"].get("epochs", 10)

    # History tracking
    history = {
        "epochs": [],  # Added epochs tracking
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

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Track epoch number (added this line)
        history["epochs"].append(epoch)

        # Training phase
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()

            # Gradient clipping (optional)
            if config["hyperparameters"].get("clip_gradients", False):
                clip_value = config["hyperparameters"].get("clip_value", 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        # Calculate average loss for the epoch
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
                    val_preds = model(x_val)
                    val_batch_loss = criterion(val_preds, y_val).item()
                    val_loss += val_batch_loss * x_val.size(0)

                    all_preds.append(val_preds.cpu().numpy())
                    all_targets.append(y_val.cpu().numpy())

            # Calculate validation loss
            val_loss = val_loss / len(val_loader.dataset)
            history["val_loss"].append(val_loss)

            # Calculate validation metrics
            all_preds = np.vstack(all_preds).flatten()
            all_targets = np.vstack(all_targets).flatten()
            metrics = compute_metrics(all_targets, all_preds)
            history["metrics"].append(metrics)

            # Log the results
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}"
            )

            # Update learning rate scheduler if needed
            if scheduler is not None and isinstance(
                scheduler, optim.lr_scheduler.ReduceLROnPlateau
            ):
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

            # Update learning rate scheduler if needed
            if scheduler is not None and not isinstance(
                scheduler, optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

        # Record epoch time
        epoch_time = time.time() - start_time
        history["epoch_times"].append(epoch_time)

    return history

