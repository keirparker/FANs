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


def prepare_data_loaders(
    t_train, data_train, config, t_val=None, data_val=None, device=None
):
    """
    Prepare data loaders for training and validation - simplified version.

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
    batch_size = config["hyperparameters"].get("batch_size", 64)
    batch_size = min(batch_size, len(t_train))
    
    # Simple approach - standardize data format regardless of input shape
    # Always make inputs and targets 2D: [batch_size, feature_dim]
    
    # Check if time series data
    is_time_series = "seq_len" in config["hyperparameters"] and "pred_len" in config["hyperparameters"]
    
    if is_time_series and len(data_train.shape) > 1 and data_train.shape[1] > 1:
        # Time series data needs special handling
        pred_len = config["hyperparameters"].get("pred_len", 24)
        
        # Split into inputs and targets
        x_data = data_train[:, :-pred_len]
        y_data = data_train[:, -pred_len:]
        
        # Convert to PyTorch tensors
        x_tensor = torch.from_numpy(x_data).float()
        y_tensor = torch.from_numpy(y_data).float()
        
        logger.info(f"Time series data: x={x_tensor.shape}, y={y_tensor.shape}")
    else:
        # Non time series - make sure we have 2D tensors
        # Always reshape to column vectors
        x_tensor = torch.from_numpy(t_train.reshape(-1, 1)).float()
        
        # Get output dimension from config to prepare targets correctly
        output_dim = config["hyperparameters"].get("output_dim", 1)
        
        if output_dim > 1:
            # If model outputs multiple values, duplicate the target to match
            y_reshaped = data_train.reshape(-1, 1)
            y_tensor = torch.from_numpy(y_reshaped).float()
            # Repeat the single value to match output_dim
            y_tensor = y_tensor.expand(-1, output_dim)
            logger.info(f"Expanded target to match output_dim={output_dim}")
        else:
            # Single output, just reshape
            y_tensor = torch.from_numpy(data_train.reshape(-1, 1)).float()
        
        logger.info(f"Standard data: x={x_tensor.shape}, y={y_tensor.shape}")

    # Move to device if specified
    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

    # Create the training loader
    train_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["hyperparameters"].get("num_workers", 0),
    )

    # Create validation loader if validation data provided
    val_loader = None
    if t_val is not None and data_val is not None:
        if is_time_series and len(data_val.shape) > 1 and data_val.shape[1] > 1:
            # Time series validation data
            x_val_data = data_val[:, :-pred_len]
            y_val_data = data_val[:, -pred_len:]
            
            x_val_tensor = torch.from_numpy(x_val_data).float()
            y_val_tensor = torch.from_numpy(y_val_data).float()
        else:
            # Non time series validation - ensure 2D tensors
            x_val_tensor = torch.from_numpy(t_val.reshape(-1, 1)).float()
            
            # Match the same output dimension as training
            if output_dim > 1:
                # If model outputs multiple values, duplicate the target to match
                y_val_reshaped = data_val.reshape(-1, 1)
                y_val_tensor = torch.from_numpy(y_val_reshaped).float()
                # Repeat the single value to match output_dim
                y_val_tensor = y_val_tensor.expand(-1, output_dim)
            else:
                # Single output, just reshape
                y_val_tensor = torch.from_numpy(data_val.reshape(-1, 1)).float()

        if device is not None:
            x_val_tensor = x_val_tensor.to(device)
            y_val_tensor = y_val_tensor.to(device)

        val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["hyperparameters"].get("num_workers", 0),
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

def train_model(model, t_train, data_train, config, device, validation_split=0.2, plot_offset_loss=False):
    """
    Enhanced training loop with loss tracking and validation.

    Args:
        model: A PyTorch model instance
        t_train: np.ndarray of shape (N,) or (N, 1) with training 'x' values
        data_train: np.ndarray of shape (N,) or (N, 1) with training 'y' values
        config: Dictionary of hyperparameters
        device: torch.device to run on
        validation_split: Fraction of data to use for validation
        plot_offset_loss: Whether to track phase offset parameter changes during training

    Returns:
        dict: Training history (losses, metrics, etc.)
    """
    # Ensure inputs have proper shape (convert 1D arrays to 2D column vectors if needed)
    if len(t_train.shape) == 1:
        logger.debug("Reshaping 1D time array to 2D column vector in train_model")
        t_train = t_train.reshape(-1, 1)
        
    if len(data_train.shape) == 1:
        logger.debug("Reshaping 1D data array to 2D column vector in train_model")
        data_train = data_train.reshape(-1, 1)
        
    logger.debug(f"Training data shapes - t_train: {t_train.shape}, data_train: {data_train.shape}")
    # Split data into training and validation if needed
    if validation_split > 0:
        val_size = int(len(t_train) * validation_split)
        indices = np.random.permutation(len(t_train))
        val_indices, train_indices = indices[:val_size], indices[val_size:]

        t_val, data_val = t_train[val_indices], data_train[val_indices]
        t_train, data_train = t_train[train_indices], data_train[train_indices]
        has_validation = True
    else:
        t_val, data_val = None, None
        has_validation = False

    # Prepare data loaders
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
    
    # Initialize offset tracking if requested
    offset_history = {}
    if plot_offset_loss:
        # Track phase offsets for models that have them
        # Look for offset parameters in the model, including nested modules
        for name, param in model.named_parameters():
            if 'offset' in name:
                offset_history[name] = []
                logger.info(f"Will track phase offset parameter: {name}")

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
            try:
                preds = model(x_batch)
                
                # Ensure predictions and targets have the same shape
                if preds.shape != y_batch.shape:
                    logger.debug(f"Shape mismatch in loss calculation: preds {preds.shape}, targets {y_batch.shape}")
                    
                    # For time series forecasting, we typically need to reshape predictions
                    # Time series targets are usually [batch_size, pred_len]
                    
                    # First, handle dimension mismatches by squeezing/unsqueezing
                    if y_batch.dim() > preds.dim():
                        y_batch = y_batch.squeeze()
                    if preds.dim() > y_batch.dim():
                        preds = preds.squeeze()
                    
                    # If predictions and targets still have different shapes
                    if preds.shape != y_batch.shape:
                        # Case 1: Target is expected to be a single value but model predicts sequence
                        if preds.shape[-1] > 1 and y_batch.shape[-1] == 1:
                            # No need to log this anymore - we're handling this automatically at the data loader level
                            # Expand target to match prediction length
                            try:
                                if y_batch.dim() == 1:  # Handle 1D target case
                                    y_batch = y_batch.unsqueeze(-1).expand(-1, preds.shape[-1])
                                else:
                                    y_batch = y_batch.expand(-1, preds.shape[-1])
                            except RuntimeError:
                                # If that fails, try reshaping predictions to match target
                                logger.debug(f"Expanding target failed, reshaping predictions to match target instead")
                                preds = preds.mean(dim=-1, keepdim=True)  # Use mean to reduce sequence to single value
                        
                        # Case 2: Model predicts single value but target is a sequence
                        elif preds.shape[-1] == 1 and y_batch.shape[-1] > 1:
                            logger.info(f"Model outputs single value but target is sequence of length {y_batch.shape[-1]} - reshaping predictions")
                            try:
                                # Expand the predictions to match target shape
                                preds = preds.expand(-1, y_batch.shape[-1])
                            except RuntimeError:
                                # If that fails, reshape target to match predictions
                                logger.info(f"Expanding predictions failed, reshaping target to match predictions instead")
                                y_batch = y_batch.mean(dim=-1, keepdim=True)  # Use mean to reduce to single value
                        
                        # Case 3: Dimensions completely different, try to adapt based on which is larger
                        else:
                            logger.warning(f"Complex shape mismatch: preds {preds.shape}, targets {y_batch.shape}. Attempting reshape.")
                            try:
                                # First try reshaping target to match predictions
                                if y_batch.dim() == 1 and preds.dim() > 1:
                                    # Handle 1D target with multidimensional predictions
                                    y_batch = y_batch.unsqueeze(-1).expand_as(preds)
                                    logger.info(f"Expanded 1D target to match predictions: {y_batch.shape}")
                                else:
                                    y_batch = y_batch.view(preds.shape)
                                    logger.info(f"Reshaped target to match predictions: {y_batch.shape}")
                            except RuntimeError:
                                try:
                                    # If that fails, try reshaping predictions to match target
                                    preds = preds.view(y_batch.shape)
                                    logger.info(f"Reshaped predictions to match target: {preds.shape}")
                                except RuntimeError:
                                    logger.warning(f"Could not match shapes: preds {preds.shape}, targets {y_batch.shape}")
                                    # As last resort, cut predictions to match first dimension of targets
                                    if preds.shape[-1] > y_batch.shape[-1]:
                                        logger.warning(f"Truncating predictions to match target dimensionality")
                                        preds = preds[..., :y_batch.shape[-1]]
                                    elif y_batch.shape[-1] > preds.shape[-1]:
                                        logger.warning(f"Truncating targets to match prediction dimensionality")
                                        y_batch = y_batch[..., :preds.shape[-1]]
                
                loss = criterion(preds, y_batch)
                loss.backward()
                
                # Gradient clipping (optional)
                if config["hyperparameters"].get("clip_gradients", False):
                    clip_value = config["hyperparameters"].get("clip_value", 1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                optimizer.step()
                running_loss += loss.item() * x_batch.size(0)
            except Exception as e:
                logger.error(f"Error in forward/backward pass: {str(e)}")
                continue

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
                    
                    # Initialize val_batch_loss to prevent UnboundLocalError
                    val_batch_loss = 0.0
                    
                    # Create copies for potential reshaping
                    y_val_calc = y_val.clone()
                    val_preds_calc = val_preds.clone()
                    
                    # Ensure predictions and targets have the same shape
                    if val_preds_calc.shape != y_val_calc.shape:
                        logger.debug(f"Shape mismatch in validation: preds {val_preds_calc.shape}, targets {y_val_calc.shape}")
                        
                        # First, handle dimension mismatches by squeezing/unsqueezing
                        if y_val_calc.dim() > val_preds_calc.dim():
                            y_val_calc = y_val_calc.squeeze()
                        if val_preds_calc.dim() > y_val_calc.dim():
                            val_preds_calc = val_preds_calc.squeeze()
                        
                        # If predictions and targets still have different shapes
                        if val_preds_calc.shape != y_val_calc.shape:
                            # Case 1: Target is expected to be a single value but model predicts sequence
                            if val_preds_calc.shape[-1] > 1 and (y_val_calc.dim() == 1 or y_val_calc.shape[-1] == 1):
                                # No need to log this anymore - handled in data loader
                                try:
                                    if y_val_calc.dim() == 1:  # Handle 1D target
                                        y_val_calc = y_val_calc.unsqueeze(-1).expand(-1, val_preds_calc.shape[-1])
                                    else:
                                        y_val_calc = y_val_calc.expand(-1, val_preds_calc.shape[-1])
                                except RuntimeError:
                                    # If expansion fails, reduce predictions to match target
                                    logger.debug("Expanding validation target failed, reducing predictions instead")
                                    val_preds_calc = val_preds_calc.mean(dim=-1, keepdim=True)
                            
                            # Case 2: Other shape mismatches, try various approaches
                            else:
                                try:
                                    # Try reshaping target to match predictions
                                    if y_val_calc.dim() == 1 and val_preds_calc.dim() > 1:
                                        # Special case for 1D targets
                                        if val_preds_calc.shape[0] == y_val_calc.shape[0]:
                                            y_val_calc = y_val_calc.unsqueeze(-1).expand_as(val_preds_calc)
                                            logger.info(f"Expanded 1D validation target: {y_val_calc.shape}")
                                    else:
                                        y_val_calc = y_val_calc.view(val_preds_calc.shape)
                                except RuntimeError:
                                    try:
                                        # Try reshaping predictions to match target
                                        val_preds_calc = val_preds_calc.view(y_val_calc.shape)
                                    except RuntimeError:
                                        logger.warning(f"Could not match shapes in validation: {val_preds_calc.shape}, {y_val_calc.shape}")
                                        
                                        # Last resort: take the mean of predictions if they have more dimensions
                                        if val_preds_calc.dim() > y_val_calc.dim() or (val_preds_calc.dim() == y_val_calc.dim() and val_preds_calc.shape[-1] > y_val_calc.shape[-1]):
                                            logger.warning(f"Using prediction mean as fallback for validation")
                                            val_preds_calc = val_preds_calc.mean(dim=-1, keepdim=(y_val_calc.dim() > 1))
                    
                    # Use the reshaped tensors for loss calculation
                    val_batch_loss = criterion(val_preds_calc, y_val_calc).item()
                    
                    val_loss += val_batch_loss * x_val.size(0)
                    
                    # Use the original tensors for metrics to maintain data consistency
                    all_preds.append(val_preds.cpu().numpy())
                    all_targets.append(y_val.cpu().numpy())

            # Calculate validation loss
            val_loss = val_loss / len(val_loader.dataset)
            history["val_loss"].append(val_loss)

            # Calculate validation metrics
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            
            # Handle shape mismatch between predictions and targets for metrics
            if all_preds.shape != all_targets.shape:
                logger.warning(f"Shape mismatch in metrics calculation: preds {all_preds.shape}, targets {all_targets.shape}")
                # If preds is a sequence but target is a single value, take the mean of predictions
                if len(all_preds.shape) > len(all_targets.shape) or all_preds.shape[1] > all_targets.shape[1]:
                    logger.info(f"Reducing predictions from shape {all_preds.shape} for metrics calculation")
                    all_preds = all_preds.mean(axis=1, keepdims=True)
                # Or if targets are more than predictions, take the mean of targets
                elif len(all_targets.shape) > len(all_preds.shape) or all_targets.shape[1] > all_preds.shape[1]:
                    logger.info(f"Reducing targets from shape {all_targets.shape} for metrics calculation")
                    all_targets = all_targets.mean(axis=1, keepdims=True)
            
            # Final flattening for metrics calculation
            all_preds = all_preds.flatten()
            all_targets = all_targets.flatten()
            
            # Double check shapes match after processing
            if all_preds.shape != all_targets.shape:
                logger.warning(f"Shapes still don't match after processing: preds {all_preds.shape}, targets {all_targets.shape}")
                # As a last resort, truncate to the same length
                min_len = min(len(all_preds), len(all_targets))
                all_preds = all_preds[:min_len]
                all_targets = all_targets[:min_len]
                
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
        
        # Track phase offset parameters if requested
        if plot_offset_loss and offset_history:
            for name, param in model.named_parameters():
                if 'offset' in name and name in offset_history:
                    # Detach from computation graph and move to CPU
                    offset_val = param.detach().cpu().numpy().copy()
                    # Handle different dimensional parameters safely
                    if offset_val.ndim == 0:
                        # Convert scalar to array
                        offset_val = np.array([offset_val])
                    elif offset_val.ndim > 1:
                        # Flatten multi-dimensional arrays
                        offset_val = offset_val.flatten()
                    
                    offset_history[name].append(offset_val)
    
    # Add offset history to the main history if we tracked any
    if plot_offset_loss and offset_history:
        history['offset_history'] = offset_history
        logger.info(f"Tracked {len(offset_history)} phase offset parameters during training")

    return history

