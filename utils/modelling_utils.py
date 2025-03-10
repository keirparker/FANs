
import mlflow
import mlflow.pytorch
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.signal_gen import get_periodic_data
from utils.data_utils import add_noise, make_sparse
from src.models import get_model_by_name



def select_device(config):
    """
    Determine which device to use based on config and hardware availability.
    If config['hyperparameters']['device'] is set to 'mps', 'cuda', or 'cpu',
    we try that. Otherwise, we pick the best available by default.
    """
    # Attempt to read device preference from config
    device_str = config["hyperparameters"].get("device", None)

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
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device (NVIDIA GPU).")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    # 3) If user explicitly asked for 'cpu', or device_str is unknown
    elif device_str == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU device as requested.")

    # 4) Otherwise, pick the best available automatically
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("No device specified; using MPS for Apple Silicon.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("No device specified; using CUDA.")
        else:
            device = torch.device("cpu")
            logger.info("No device specified; using CPU.")

    return device


def train_model(model, t_train, data_train, config, device, validation_split=0.1):
    """
    Training loop with device support and validation metrics.
    Args:
        model: A PyTorch model instance.
        t_train: np.ndarray with training 'x' values.
        data_train: np.ndarray with training 'y' values.
        config: Dictionary of hyperparameters.
        device: torch.device to run on (cpu, cuda, or mps).
        validation_split: Fraction of training data to use for validation.

    Returns:
        history (dict): A dictionary containing training metrics across epochs.
    """
    # Check if we're dealing with time series data
    is_time_series = "seq_len" in config["hyperparameters"] and "pred_len" in config["hyperparameters"]
    
    if is_time_series:
        logger.info("Detecting time series data format for training")
        pred_len = config["hyperparameters"].get("pred_len", 24)
        
        # For time series data, separate inputs and targets
        x_data = data_train[:, :-pred_len]  # All columns except last pred_len
        y_data = data_train[:, -pred_len:]  # Only last pred_len columns
        
        # Convert to PyTorch tensors
        x_tensor = torch.from_numpy(x_data).float().to(device)
        y_tensor = torch.from_numpy(y_data).float().to(device)
    else:
        # Standard data format
        x_tensor = torch.from_numpy(t_train).float().unsqueeze(-1).to(device)
        y_tensor = torch.from_numpy(data_train).float().unsqueeze(-1).to(device)

    # Simple MSE loss
    criterion = nn.MSELoss()

    # Basic optimizer (e.g. Adam)
    lr = config["hyperparameters"].get("lr", 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = config["hyperparameters"].get("epochs", 10)
    batch_size = min(64, len(x_tensor))  # Use reasonable batch size
    
    # Split data into training and validation sets
    dataset_size = len(x_tensor)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    # Shuffle indices
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create dataset and dataloaders
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler
    )
    
    # Initialize history tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": [],
        "metrics": [],
        "epoch_times": [],
        "epoch_start_time": time.time()  # Track training start time for efficiency metrics
    }

    model.train()
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_batches = 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            try:
                preds = model(x_batch)  # Forward pass
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x_batch.size(0)
                train_batches += 1
            except RuntimeError as e:
                logger.error(f"Error during training: {e}")
                logger.error(f"Input shape: {x_batch.shape}, Target shape: {y_batch.shape}")
                raise
                
        # Calculate epoch loss
        epoch_train_loss = running_loss / (train_batches * batch_size) if train_batches > 0 else float('inf')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                try:
                    val_preds = model(x_val)
                    batch_val_loss = criterion(val_preds, y_val).item()
                    val_loss += batch_val_loss * x_val.size(0)
                    val_batches += 1
                except RuntimeError as e:
                    logger.error(f"Error during validation: {e}")
                    logger.error(f"Input shape: {x_val.shape}, Target shape: {y_val.shape}")
                    continue
                    
        # Calculate validation loss
        epoch_val_loss = val_loss / (val_batches * batch_size) if val_batches > 0 else float('inf')
        
        # Log progress
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        # Track epoch time
        current_time = time.time()
        epoch_time = current_time - history.get('epoch_start_time', current_time)
        history['epoch_times'].append(epoch_time)
        history['epoch_start_time'] = current_time
        
        # Store metrics
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["epochs"].append(epoch + 1)
        history["metrics"].append({"mse": epoch_val_loss})

    return history  # Return the training history



def evaluate_model(model, t_test, data_test, device, config=None):
    """
    Evaluate the model on test data, returning evaluation metrics and predictions.

    Args:
        model: A PyTorch model instance.
        t_test: np.ndarray of shape (N,) with test 'x' values.
        data_test: np.ndarray of shape (N,) with test 'y' values.
        device: torch.device to run on (cpu, cuda, or mps).
        config: Optional configuration dictionary for time series data handling.

    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
        predictions (np.ndarray): Model predictions, shape (N,).
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.metrics import (
        mean_squared_error,
        r2_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
    )
    from loguru import logger

    criterion = nn.MSELoss()

    # Check if this is a time series forecasting task
    is_time_series = config is not None and "seq_len" in config["hyperparameters"] and "pred_len" in config["hyperparameters"]
    
    if is_time_series:
        # For time series data, we need to separate inputs and targets
        pred_len = config["hyperparameters"].get("pred_len", 24)
        
        # The data_test has combined [inputs, targets] format
        # The last pred_len columns are the targets, the rest are inputs
        x_data = data_test[:, :-pred_len]  # Inputs (all except the last pred_len columns)
        y_data = data_test[:, -pred_len:]  # Targets (just the last pred_len columns)
        
        logger.info(f"Time series test data shapes - X: {x_data.shape}, Y: {y_data.shape}")
        
        # Convert to PyTorch tensors
        x_tensor = torch.from_numpy(x_data).float().to(device)
        y_tensor = torch.from_numpy(y_data).float().to(device)
    else:
        # For regular data, use original format
        # Convert to PyTorch tensors, but don't unsqueeze if already 2D
        if len(t_test.shape) == 1:
            x_tensor = torch.from_numpy(t_test).float().unsqueeze(-1).to(device)
        else:
            x_tensor = torch.from_numpy(t_test).float().to(device)
            
        if len(data_test.shape) == 1:
            y_tensor = torch.from_numpy(data_test).float().unsqueeze(-1).to(device)
        else:
            y_tensor = torch.from_numpy(data_test).float().to(device)

    model.eval()
    
    # Process in batches to avoid memory issues
    batch_size = 64  # Use a reasonable batch size
    
    if is_time_series:
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                batch_preds = model(x_batch)
                batch_loss = criterion(batch_preds, y_batch).item()
                
                total_loss += batch_loss * len(x_batch)
                total_samples += len(x_batch)
                
                # Store predictions
                all_preds.append(batch_preds.cpu().numpy())
                
            # Calculate average MSE
            mse_loss = total_loss / total_samples
            
            # Combine predictions
            preds_np = np.vstack(all_preds)
            target_np = y_data
    else:
        # For non-time series data
        with torch.no_grad():
            try:
                preds = model(x_tensor)
            except Exception as e:
                logger.error(f"Error during model evaluation: {str(e)}")
                # Return empty predictions and default metrics
                default_metrics = {"mse": float('inf'), "rmse": float('inf'), 
                                 "mae": float('inf'), "r2": -float('inf')}
                return default_metrics, np.zeros((len(x_tensor), 1))
            
            # Ensure predictions and targets have the same shape before computing loss
            if preds.shape != y_tensor.shape:
                logger.debug(f"Shape mismatch in evaluation: preds {preds.shape}, targets {y_tensor.shape}")
                
                # Create copies for reshaping
                y_calc = y_tensor.clone()
                preds_calc = preds.clone()
                
                # First, handle dimension mismatches by squeezing/unsqueezing
                if y_calc.dim() > preds_calc.dim():
                    y_calc = y_calc.squeeze()
                if preds_calc.dim() > y_calc.dim():
                    preds_calc = preds_calc.squeeze()
                
                # If predictions and targets still have different shapes,
                # and predictions have only one output but targets have many (time series case)
                if preds_calc.shape[-1] == 1 and y_calc.shape[-1] > 1:
                    try:
                        # Expand the predictions to match target shape
                        preds_calc = preds_calc.expand(-1, y_calc.shape[-1])
                        logger.debug(f"Output dim mismatch - expanded predictions from 1 to {y_calc.shape[-1]}")
                    except RuntimeError:
                        logger.debug(f"Could not expand predictions to match targets")
                # Otherwise try to reshape one to match the other
                elif preds_calc.shape != y_calc.shape:
                    try:
                        y_calc = y_calc.view(preds_calc.shape)
                    except RuntimeError:
                        try:
                            preds_calc = preds_calc.view(y_calc.shape)
                        except RuntimeError:
                            # Manually expand the targets to match outputs
                            if preds_calc.shape[-1] > 1 and y_calc.shape[-1] == 1:
                                logger.debug(f"Expanding targets from 1 to {preds_calc.shape[-1]} dimensions")
                                try:
                                    y_calc = y_calc.expand(-1, preds_calc.shape[-1])
                                except RuntimeError:
                                    logger.debug(f"Could not match shapes in evaluation: {preds_calc.shape}, {y_calc.shape}")
                            else:
                                logger.debug(f"Could not match shapes in evaluation: {preds_calc.shape}, {y_calc.shape}")
                
                # Use the properly shaped tensors for loss calculation
                # Final check if shapes still don't match
                if preds_calc.shape != y_calc.shape:
                    logger.debug(f"Final attempt to match shapes: preds {preds_calc.shape}, targets {y_calc.shape}")
                    # Special case for multi-output models with single-value targets
                    if preds_calc.shape[0] == y_calc.shape[0] and preds_calc.shape[1] > 1 and (y_calc.dim() == 1 or y_calc.shape[1] == 1):
                        if y_calc.dim() == 1:
                            y_calc = y_calc.unsqueeze(-1)
                        try:
                            # Try expanding the target
                            y_calc = y_calc.expand(-1, preds_calc.shape[1])
                            logger.debug(f"Successfully expanded targets to {y_calc.shape}")
                        except RuntimeError:
                            # If expansion fails, reshape predictions to match target
                            logger.debug(f"Expanding targets failed, reshaping predictions instead")
                            preds_calc = preds_calc.mean(dim=-1, keepdim=True)
                
                mse_loss = criterion(preds_calc, y_calc).item()
            else:
                mse_loss = criterion(preds, y_tensor).item()
                
            # Convert preds back to numpy (on CPU)
            preds_np = preds.cpu().numpy()
            target_np = data_test
            
            # Handle dimension mismatch for metrics calculation
            if preds_np.shape != target_np.shape:
                logger.debug(f"Shape mismatch for metrics calculation: preds {preds_np.shape}, targets {target_np.shape}")
                # Special case: if preds is [batch_size, output_dim] and target is [batch_size, 1],
                # expand the target to match prediction dimension
                if len(preds_np.shape) == 2 and len(target_np.shape) == 2:
                    if preds_np.shape[0] == target_np.shape[0] and preds_np.shape[1] > 1 and target_np.shape[1] == 1:
                        logger.debug(f"Expanding targets to match prediction dimension: {preds_np.shape[1]}")
                        target_np = np.repeat(target_np, preds_np.shape[1], axis=1)
                
                # Case 1: Predictions have more dimensions than targets
                if preds_np.ndim > target_np.ndim:
                    if preds_np.shape[0] == target_np.shape[0]:
                        # If first dimension matches (batch size), average across other dimensions
                        logger.debug(f"Averaging predictions across output dimensions to match target shape")
                        preds_np = np.mean(preds_np, axis=tuple(range(1, preds_np.ndim)))
                
                # Case 2: Targets have more dimensions than predictions
                elif target_np.ndim > preds_np.ndim:
                    if target_np.shape[0] == preds_np.shape[0]:
                        # If first dimension matches, average targets across other dimensions
                        logger.debug(f"Averaging targets across output dimensions to match prediction shape")
                        target_np = np.mean(target_np, axis=tuple(range(1, target_np.ndim)))
                
                # Check shapes after first attempt at reshaping
                if preds_np.shape != target_np.shape:
                    # Case 3: Same number of dimensions but different shapes
                    if preds_np.ndim == target_np.ndim:
                        # If batch dimension matches but other dimensions don't
                        if preds_np.shape[0] == target_np.shape[0]:
                            if preds_np.shape[-1] == 1 and target_np.shape[-1] > 1:
                                # Expand predictions to match targets
                                logger.debug(f"Expanding predictions to match target shape")
                                preds_np = np.tile(preds_np, (1, target_np.shape[-1]))
                            elif preds_np.shape[-1] > 1 and target_np.shape[-1] == 1:
                                # Resize targets to match predictions
                                logger.debug(f"Expanding targets to match prediction shape")
                                target_np = np.tile(target_np, (1, preds_np.shape[-1]))
                            else:
                                # Truncate to the smaller of the two shapes
                                min_shape = min(preds_np.shape[-1], target_np.shape[-1])
                                logger.debug(f"Truncating both arrays to common size {min_shape}")
                                preds_np = preds_np[..., :min_shape]
                                target_np = target_np[..., :min_shape]
                
                # Final check - if still mismatched, reshape one of them
                if preds_np.shape != target_np.shape:
                    logger.debug(f"Final reshape attempt - taking first value only")
                    # Last resort: take only first value from the array with more values
                    if preds_np.size > target_np.size:
                        preds_np = preds_np.flatten()[:target_np.size].reshape(target_np.shape)
                    else:
                        target_np = target_np.flatten()[:preds_np.size].reshape(preds_np.shape)

    # Calculate metrics
    try:
        mse = mean_squared_error(target_np, preds_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_np, preds_np)
        r2 = r2_score(target_np, preds_np)
    except ValueError as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        logger.error(f"Shapes after processing: preds {preds_np.shape}, targets {target_np.shape}")
        # Return default metrics in case of error
        return {"mse": float('inf'), "rmse": float('inf'), "mae": float('inf'), 
                "r2": -float('inf'), "mape": float('inf')}, preds_np

    # Calculate MAPE, handling potential division by zero
    try:
        mape = mean_absolute_percentage_error(target_np, preds_np) * 100
    except:
        # Handle cases where actual values contain zeros
        mape = (
            np.mean(
                np.abs((target_np - preds_np) / np.maximum(np.abs(target_np), 1e-10))
            )
            * 100
        )

    # Create metrics dictionary
    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

    return metrics, preds_np


def run_experiment(model_name, dataset_type, data_version, config):
    """
    Runs a single experiment:
    1) Generate data
    2) Possibly add noise or sparsify
    3) Train a PyTorch model
    4) Evaluate & plot with MLflow
    """
    mlflow.start_run()

    # Determine which device to use
    device = select_device(config)

    # (1) Generate the base dataset + retrieve underlying function
    # We assume get_periodic_data returns (t_train, data_train, t_test, data_test, data_config, true_func)
    (
        t_train,
        data_train,
        t_test,
        data_test,
        data_config,
        true_func
    ) = get_periodic_data(
        periodic_type=dataset_type,
        num_train_samples=config["hyperparameters"]["num_samples"],
        num_test_samples=config["hyperparameters"]["test_samples"],
    )

    # (2) Apply data transformation if needed
    if data_version == "original":
        logger.info("Using original data (no transformation).")
    elif data_version == "noisy":
        noise_level = config["hyperparameters"]["noise_level"]
        logger.info(f"Applying noise (level={noise_level}) to training & test data.")
        data_train = add_noise(data_train, noise_level=noise_level)
        data_test = add_noise(data_test, noise_level=noise_level)
    elif data_version == "sparse":
        sparsity_factor = config["hyperparameters"]["sparsity_factor"]
        logger.info(f"Making data sparse (factor={sparsity_factor}).")
        data_train, idx_train = make_sparse(data_train, sparsity_factor=sparsity_factor)
        data_test, idx_test = make_sparse(data_test, sparsity_factor=sparsity_factor)
        # Also shrink time arrays
        t_train = t_train[idx_train]
        t_test = t_test[idx_test]
    else:
        logger.error(f"Unknown data version: {data_version}")
        raise ValueError("Invalid data version")

    # (3) Retrieve & train the model
    model = get_model_by_name(model_name)
    model.to(device)  # Move model to device
    final_train_loss = train_model(model, t_train, data_train, config, device)

    # (4) Evaluate on test set
    test_loss, preds_test = evaluate_model(model, t_test, data_test, device)

    # (5) Plot everything
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))

    # Underlying "true" function (red dashed)
    if true_func is not None:
        t_dense = np.linspace(min(t_test), max(t_test), 2000)
        y_dense = true_func(t_dense)
        plt.plot(t_dense, y_dense, "r--", label="True Function", alpha=0.7)

    # Training data (blue dots)
    plt.scatter(t_train, data_train, color="blue", s=10, alpha=0.5, label="Train Data")

    # Test data (green dots)
    plt.scatter(t_test, data_test, color="green", s=10, alpha=0.5, label="Test Data")

    # Model predictions (magenta line)
    sort_idx = np.argsort(t_test)
    plt.plot(t_test[sort_idx], preds_test[sort_idx], "m-", label="Model Prediction")

    plt.title(f"{model_name} | {dataset_type} ({data_version})")
    plt.legend()
    plot_path = f"plots/{model_name}_{dataset_type}_{data_version}.png"
    plt.savefig(plot_path)
    plt.close()

    # (6) Log metrics & artifacts with MLflow

    mlflow.log_param("model", model_name)
    mlflow.log_param("dataset_type", dataset_type)
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("num_samples", config["hyperparameters"]["num_samples"])
    mlflow.log_param("epochs", config["hyperparameters"]["epochs"])
    mlflow.log_param("lr", config["hyperparameters"]["lr"])
    mlflow.log_param("device", device.type)

    mlflow.log_metric("train_loss", final_train_loss)
    mlflow.log_metric("test_loss", test_loss)

    # Log the plot
    mlflow.log_artifact(plot_path)

        # (Optional) Save the trained model to MLflow
        # mlflow.pytorch.log_model(model, artifact_path="model")

    logger.info(
        f"Completed experiment: model={model_name}, dataset={dataset_type}, version={data_version}. "
        f"Train Loss={final_train_loss:.4f}, Test Loss={test_loss:.4f}"
    )



