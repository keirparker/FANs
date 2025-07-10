#!/usr/bin/env python
"""
Time Series Forecasting Runner
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import mlflow
from loguru import logger
import logging
import time
from datetime import datetime
import yaml
import json
import ray
from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
# Try to import MLflow integration, fallback if not available
try:
    from ray.tune.integration.mlflow import MLflowLoggerCallback
except ImportError:
    print("Warning: MLflowLoggerCallback not available, using basic logging")
    MLflowLoggerCallback = None
from functools import partial
import psutil
import platform

# Add the project root to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports from time-series directory
from models import get_model_by_name, list_available_models
from utils.config_utils import setup_environment
from utils.device_utils import select_device
from utils.training_utils import create_optimizer, create_scheduler
from utils.evaluation_utils import plot_losses_by_epoch_comparison
from utils.convergence_utils import calculate_convergence_speed, calculate_training_efficiency, compare_convergence


def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(
        f"{log_dir}/ts_experiments_{time.strftime('%Y%m%d-%H%M%S')}.log",
        rotation="50 MB",
        retention="5 days",
        compression="zip",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(
        lambda msg: print(msg),
        level="INFO",
        format="{level} | <blue>{time:HH:mm:ss}</blue> | <level>{message}</level>",
    )
    
    logger.info(f"Logger configured to output in {log_dir}")


def load_config(config_path="config.yml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def init_ray_for_mac():
    num_cpus = psutil.cpu_count(logical=True)
    reserved_cpus = 2
    ray_cpus = max(1, num_cpus - reserved_cpus)
    
    if not ray.is_initialized():
        try:
            ray.init(
                num_cpus=ray_cpus,
                include_dashboard=False,  # Disable dashboard for stability
                ignore_reinit_error=True,
                _temp_dir="/tmp/ray_temp",  # Prevent permissions issues on macOS
                _system_config={
                    "worker_register_timeout_seconds": 60,
                    # Removed problematic parameter: raylet_startup_token_refresh_ms
                    "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}}',
                    "max_io_workers": 4  # Reduce I/O worker threads for Mac
                },
                logging_level=logging.WARNING
            )
            logger.info(f"Ray initialized with {ray_cpus} CPUs on macOS {platform.mac_ver()[0]} {platform.processor()}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ray: {e}. Will continue without parallel processing.")
            return False
    
    return True


def load_dataset(dataset_name, config):
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Generate synthetic time series data for demonstration
    logger.info(f"Generating synthetic time series data for {dataset_name}")
    
    # Create synthetic multivariate time series data
    import numpy as np
    import torch
    
    # Configuration for synthetic data
    seq_length = 10000  # Length of time series
    n_features = 7      # Number of features (typical for ETT datasets)
    
    # Generate time-based features with different periodicities
    t = np.linspace(0, 100, seq_length)
    
    # Create synthetic multivariate time series with realistic patterns
    data = np.zeros((seq_length, n_features))
    
    # Feature 1: Main trend with daily pattern
    data[:, 0] = np.sin(2 * np.pi * t / 24) + 0.1 * np.sin(2 * np.pi * t / 12) + 0.01 * t
    
    # Feature 2: Weekly pattern
    data[:, 1] = np.cos(2 * np.pi * t / (24 * 7)) + 0.5 * np.sin(2 * np.pi * t / 24)
    
    # Feature 3: Temperature-like pattern
    data[:, 2] = 20 + 10 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / (24 * 365))
    
    # Feature 4: Load pattern
    data[:, 3] = 50 + 30 * np.sin(2 * np.pi * t / 24 + np.pi/4) + 10 * np.random.randn(seq_length) * 0.1
    
    # Features 5-7: Additional correlated features
    for i in range(4, n_features):
        data[:, i] = 0.7 * data[:, 0] + 0.3 * data[:, 1] + 0.1 * np.random.randn(seq_length)
    
    # Add some noise to all features
    data += 0.01 * np.random.randn(seq_length, n_features)
    
    # Convert to torch tensor
    data = torch.FloatTensor(data)
    
    logger.info(f"Generated synthetic {dataset_name} with shape {data.shape}")
    
    return data


def create_time_series_loaders(data, lookback, horizon, stride, batch_size, num_workers, pin_memory, prefetch_factor, persistent_workers):
    """Create train, validation, and test data loaders for time series data."""
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    
    # Split data into train/val/test
    n_samples = len(data)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    def create_sequences(data, lookback, horizon, stride=1):
        """Create input-output sequences from time series data."""
        X, y = [], []
        for i in range(0, len(data) - lookback - horizon + 1, stride):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback:i + lookback + horizon])
        return torch.stack(X), torch.stack(y)
    
    # Create sequences for each split
    X_train, y_train = create_sequences(train_data, lookback, horizon, stride)
    X_val, y_val = create_sequences(val_data, lookback, horizon, stride)
    X_test, y_test = create_sequences(test_data, lookback, horizon, stride)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0
    )
    
    logger.info(f"Created data loaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def create_model(model_name, model_params, device):
    """Create and initialize a model from the model registry."""
    try:
        # Create model using the registry (which handles instantiation)
        model = get_model_by_name(model_name, **model_params)
        
        # Move to device
        model = model.to(device)
        
        logger.info(f"Created {model_name} with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        raise


@ray.remote
def train_model_distributed(model_params, train_loader, val_loader, config, device_type='cpu'):
    device = torch.device(device_type)
    
    model_name = model_params.pop("model_name")
    model = get_model_by_name(model_name, **model_params)
    model = model.to(device)
    
    if config["hyperparameters"].get("use_compile", False) and hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        try:
            logger.info("Using torch.compile for model optimization")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}, continuing with standard model")
    
    if config["hyperparameters"].get("use_checkpoint", False):
        if hasattr(model, "transformer_encoder") and hasattr(model.transformer_encoder, "use_checkpointing"):
            logger.info("Enabling gradient checkpointing")
            model.transformer_encoder.use_checkpointing = True
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    num_epochs = config["hyperparameters"]["epochs"]
    log_interval = config.get("logging", {}).get("log_interval", 10)
    
    grad_accum_steps = config["hyperparameters"].get("grad_accum_steps", 1)
    if grad_accum_steps > 1:
        logger.info(f"Using gradient accumulation with {grad_accum_steps} steps")
        
    use_amp = config["hyperparameters"].get("use_amp", False)
    use_amp = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None
    
    if use_amp:
        logger.info(f"Using Automatic Mixed Precision (AMP) for {device.type}")
    elif config["hyperparameters"].get("use_amp", False):
        logger.info(f"Automatic Mixed Precision (AMP) not fully supported on {device.type}, disabled")
    
    criterion = torch.nn.MSELoss()
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": [],
        "lr": [],
        "training_time": []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        epoch_start_time = time.time()
        
        total_batches = len(train_loader)
        batch_log_interval = max(1, total_batches // 4)
        
        warmup_epochs = min(3, num_epochs // 3)
        if epoch <= warmup_epochs and hasattr(optimizer, 'param_groups'):
            scale = min(1.0, epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config["hyperparameters"]["lr"] * scale
        
        effective_batch_size = grad_accum_steps * train_loader.batch_size
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % grad_accum_steps == 0:
                optimizer.zero_grad()
            
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            if use_amp and scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    output = model(data)
                    loss = criterion(output, target) / grad_accum_steps
                    
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target) / grad_accum_steps
                loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            
            train_loss += loss.item() * grad_accum_steps
            
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_start_time = time.time()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                if use_amp:
                    with torch.amp.autocast(device_type=device.type):
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    
                val_loss += loss.item()
                
        val_time = time.time() - val_start_time
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_total_time = time.time() - epoch_start_time
        history["training_time"].append(epoch_total_time)
        
        if scheduler is not None:
            scheduler.step()
            
        if hasattr(model, 'update_phase_coefficients'):
            model.update_phase_coefficients(epoch)
            
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epochs"].append(epoch)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        samples_per_second = len(train_loader.dataset) / epoch_time
        batches_per_second = len(train_loader) / epoch_time
            
        min_improvement = config.get("min_improvement", 0.005)
        
        if avg_val_loss < best_val_loss:
            improvement = (best_val_loss - avg_val_loss) / best_val_loss if best_val_loss != float('inf') else 1.0
            improvement_pct = improvement * 100
            
            if improvement >= min_improvement or best_val_loss == float('inf'):
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
            else:
                best_val_loss = avg_val_loss
            
        session.report({"val_loss": avg_val_loss, "train_loss": avg_train_loss, "epoch": epoch})
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history, best_model_state, model


def train_model(model, train_loader, val_loader, config, device):
    model = model.to(device)
    
    if config["hyperparameters"].get("use_compile", False) and hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        try:
            logger.info("Using torch.compile for model optimization")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}, continuing with standard model")
    
    if config["hyperparameters"].get("use_checkpoint", False):
        if hasattr(model, "transformer_encoder") and hasattr(model.transformer_encoder, "use_checkpointing"):
            logger.info("Enabling gradient checkpointing")
            model.transformer_encoder.use_checkpointing = True
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    num_epochs = config["hyperparameters"]["epochs"]
    log_interval = config.get("logging", {}).get("log_interval", 10)
    
    grad_accum_steps = config["hyperparameters"].get("grad_accum_steps", 1)
    if grad_accum_steps > 1:
        logger.info(f"Using gradient accumulation with {grad_accum_steps} steps")
        
    use_amp = config["hyperparameters"].get("use_amp", False)
    use_amp = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None
    
    if use_amp:
        logger.info(f"Using Automatic Mixed Precision (AMP) for {device.type}")
    elif config["hyperparameters"].get("use_amp", False):
        logger.info(f"Automatic Mixed Precision (AMP) not fully supported on {device.type}, disabled")
    
    criterion = torch.nn.MSELoss()
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": [],
        "lr": [],
        "training_time": []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        epoch_start_time = time.time()
        
        total_batches = len(train_loader)
        batch_log_interval = max(1, total_batches // 4)
        logger.info(f"Epoch {epoch}/{num_epochs} - Starting training ({total_batches} batches)")
        
        warmup_epochs = min(3, num_epochs // 3)
        if epoch <= warmup_epochs and hasattr(optimizer, 'param_groups'):
            scale = min(1.0, epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config["hyperparameters"]["lr"] * scale
            logger.info(f"Warmup LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        effective_batch_size = grad_accum_steps * train_loader.batch_size
        if grad_accum_steps > 1:
            logger.info(f"Effective batch size: {effective_batch_size} (accumulation: {grad_accum_steps} × {train_loader.batch_size})")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if not hasattr(model, '_batch_start_time'):
                model._batch_start_time = time.time()
                model._last_batch_idx = 0
            
            if batch_idx % batch_log_interval == 0 or batch_idx == total_batches - 1:
                current_time = time.time()
                progress_pct = (batch_idx + 1) / total_batches * 100
                
                time_elapsed = current_time - epoch_start_time
                overall_time_per_batch = time_elapsed / (batch_idx + 1) 
                
                batches_since_last = batch_idx - model._last_batch_idx + 1
                time_since_last = current_time - model._batch_start_time
                recent_time_per_batch = time_since_last / batches_since_last if batches_since_last > 0 else 0
                
                model._batch_start_time = current_time
                model._last_batch_idx = batch_idx + 1
                
                eta_seconds = recent_time_per_batch * (total_batches - batch_idx - 1)
                
                if eta_seconds > 60:
                    eta = f"{eta_seconds/60:.1f} min"
                else:
                    eta = f"{eta_seconds:.1f} sec"
                    
                logger.info(f"Epoch {epoch}/{num_epochs} - Batch {batch_idx+1}/{total_batches} "
                          f"({progress_pct:.1f}%) - Current rate: {recent_time_per_batch:.3f}s/batch - "
                          f"Avg rate: {overall_time_per_batch:.3f}s/batch - ETA: {eta}")
            
            if batch_idx % grad_accum_steps == 0:
                optimizer.zero_grad()
            
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            if use_amp and scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    output = model(data)
                    loss = criterion(output, target) / grad_accum_steps
                    
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target) / grad_accum_steps
                loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            
            train_loss += loss.item() * grad_accum_steps
            
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_start_time = time.time()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                if use_amp:
                    with torch.amp.autocast(device_type=device.type):
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    
                val_loss += loss.item()
                
        val_time = time.time() - val_start_time
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_total_time = time.time() - epoch_start_time
        history["training_time"].append(epoch_total_time)
        
        if scheduler is not None:
            scheduler.step()
            
        if hasattr(model, 'update_phase_coefficients'):
            model.update_phase_coefficients(epoch)
            logger.info(f"Updated phase coefficients for HybridPhaseFAN model (epoch {epoch})")
            
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epochs"].append(epoch)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        samples_per_second = len(train_loader.dataset) / epoch_time
        batches_per_second = len(train_loader) / epoch_time
        
        logger.info(f"Epoch {epoch}/{num_epochs} completed in {epoch_time:.2f}s (train) + {val_time:.2f}s (val) = {epoch_total_time:.2f}s: "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                  f"Speed: {batches_per_second:.2f} batches/s ({samples_per_second:.1f} samples/s)")
            
        min_improvement = config.get("min_improvement", 0.005)
        
        if avg_val_loss < best_val_loss:
            improvement = (best_val_loss - avg_val_loss) / best_val_loss if best_val_loss != float('inf') else 1.0
            improvement_pct = improvement * 100
            
            if improvement >= min_improvement or best_val_loss == float('inf'):
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                logger.info(f"Epoch {epoch}: New best model with val_loss {best_val_loss:.6f} ({improvement_pct:.2f}% improvement)")
            else:
                best_val_loss = avg_val_loss
                logger.info(f"Epoch {epoch}: Small improvement ({improvement_pct:.2f}% < {min_improvement*100:.1f}%) - not saving model")
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history, best_model_state


def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    test_mae = 0
    criterion = torch.nn.MSELoss()
    mae_criterion = torch.nn.L1Loss()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            test_mae += mae_criterion(output, target).item()
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_mae = test_mae / len(test_loader)
    rmse = np.sqrt(avg_test_loss)
    
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    logger.info(f"Test metrics - MSE: {avg_test_loss:.6f}, RMSE: {rmse:.6f}, MAE: {avg_test_mae:.6f}")
    
    return {
        "mse": avg_test_loss,
        "rmse": rmse,
        "mae": avg_test_mae
    }, predictions, targets


def tune_hyperparameters(model_name, dataset_name, config, experiment_id):
    device, _ = select_device(config)
    device_type = device.type
    
    data = load_dataset(dataset_name, config)
    
    if "dataset_config" in config and dataset_name in config["dataset_config"]:
        dataset_config = config["dataset_config"][dataset_name]
        lookback = dataset_config.get("lookback", config["hyperparameters"].get("lookback", 96))
        horizon = dataset_config.get("horizon", config["hyperparameters"].get("horizon", 24))
        config_input_dim = dataset_config.get("input_dim", None)
    else:
        lookback = config["hyperparameters"].get("lookback", 96)
        horizon = config["hyperparameters"].get("horizon", 24)
    
    batch_size = config["hyperparameters"]["batch_size"]
    if "dataset_config" in config and dataset_name.lower() in config["dataset_config"]:
        if "batch_size" in config["dataset_config"][dataset_name.lower()]:
            batch_size = config["dataset_config"][dataset_name.lower()]["batch_size"]
    
    actual_input_dim = data.shape[1] if len(data.shape) > 1 else 1
    
    train_loader, val_loader, test_loader = create_time_series_loaders(
        data,
        lookback=lookback,
        horizon=horizon,
        stride=config["hyperparameters"]["stride"],
        batch_size=batch_size,
        num_workers=config["hyperparameters"]["num_workers"],
        pin_memory=config["hyperparameters"].get("pin_memory", True),
        prefetch_factor=config["hyperparameters"].get("prefetch_factor", 2),
        persistent_workers=config["hyperparameters"].get("persistent_workers", True)
    )
    
    param_space = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "dropout": tune.uniform(0.0, 0.5),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "n_layers": tune.choice([2, 3, 4, 6]),
    }
    
    if "Transformer" in model_name:
        param_space.update({
            "n_heads": tune.choice([4, 8]),
            "d_model": tune.choice([128, 256, 512])
        })
    
    def train_model_tune(config_hp):
        model_params = {
            "model_name": model_name,
            "input_dim": actual_input_dim,
            "output_dim": actual_input_dim,
            "horizon": horizon,
            "dropout": config_hp["dropout"]
        }
        
        if "InterferenceFANForecaster" in model_name:
            model_params.update({
                "hidden_dim": config_hp["hidden_dim"],
                "n_interference_layers": config_hp["n_layers"],
            })
        elif "FANForecaster" in model_name or "FANGatedForecaster" in model_name or "HybridPhaseFANForecaster" in model_name:
            model_params.update({
                "hidden_dim": config_hp["hidden_dim"],
                "n_fan_layers": config_hp["n_layers"],
            })
            
            if "HybridPhaseFANForecaster" in model_name:
                model_params.update({
                    "phase_decay_epochs": config["hyperparameters"]["phase_decay_epochs"],
                })
        elif "LSTMForecaster" in model_name:
            model_params.update({
                "hidden_dim": config_hp["hidden_dim"],
                "num_layers": config_hp["n_layers"],
            })
        elif "FANTransformerForecaster" in model_name or "PhaseOffsetTransformerForecaster" in model_name or "HybridPhaseFANTransformerForecaster" in model_name:
            use_checkpointing = config["hyperparameters"].get("use_checkpoint", False)
            
            model_params.update({
                "d_model": config_hp["d_model"],
                "fan_dim": config_hp["hidden_dim"],
                "nhead": config_hp["n_heads"],
                "num_layers": config_hp["n_layers"],
                "use_checkpointing": use_checkpointing,
            })
            
            if "HybridPhaseFANTransformerForecaster" in model_name:
                model_params.update({
                    "phase_decay_epochs": config["hyperparameters"]["phase_decay_epochs"],
                })
        elif "TransformerForecaster" in model_name:
            model_params.update({
                "hidden_dim": config_hp["d_model"],
                "nhead": config_hp["n_heads"],
                "num_layers": config_hp["n_layers"],
            })
        
        tune_config = config.copy()
        tune_config["hyperparameters"] = tune_config["hyperparameters"].copy()
        tune_config["hyperparameters"]["lr"] = config_hp["lr"]
        tune_config["hyperparameters"]["dropout"] = config_hp["dropout"]
        tune_config["hyperparameters"]["hidden_dim"] = config_hp["hidden_dim"]
        tune_config["hyperparameters"]["n_layers"] = config_hp["n_layers"]
        
        if "Transformer" in model_name:
            tune_config["hyperparameters"]["n_heads"] = config_hp["n_heads"]
            tune_config["hyperparameters"]["d_model"] = config_hp["d_model"]
        
        tune_config["hyperparameters"]["epochs"] = min(config["hyperparameters"]["epochs"], 30)
        
        history, best_model_state, model = train_model_distributed.remote(
            model_params, train_loader, val_loader, tune_config, device_type
        ).get()
        
        best_val_loss = min(history["val_loss"])
        return {"best_val_loss": best_val_loss}
    
    search_alg = OptunaSearch(
        metric="best_val_loss", 
        mode="min",
        points_to_evaluate=[{
            "lr": config["hyperparameters"]["lr"],
            "dropout": config["hyperparameters"]["dropout"],
            "hidden_dim": config["hyperparameters"]["hidden_dim"],
            "n_layers": config["hyperparameters"]["n_layers"],
            "n_heads": config["hyperparameters"].get("n_heads", 8),
            "d_model": config["hyperparameters"].get("d_model", 256)
        }]
    )
    
    mlflow_callback = MLflowLoggerCallback(
        experiment_name=f"Tune_{model_name}_{dataset_name}",
        tracking_uri=mlflow.get_tracking_uri()
    )
    
    tuner = tune.Tuner(
        train_model_tune,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="best_val_loss",
            mode="min",
            num_samples=config["hyperparameters"].get("tune_samples", 10),
            max_concurrent_trials=config["hyperparameters"].get("tune_concurrency", 2),
            search_alg=search_alg
        ),
        run_config=ray.air.RunConfig(
            name=f"tune_{model_name}_{dataset_name}",
            callbacks=[mlflow_callback]
        )
    )
    
    results = tuner.fit()
    best_trial = results.get_best_result(metric="best_val_loss", mode="min")
    
    logger.info(f"Best hyperparameters found: {best_trial.config}")
    logger.info(f"Best validation loss: {best_trial.metrics['best_val_loss']:.6f}")
    
    return best_trial.config


def run_experiment_sequential(model_name, dataset_name, config, experiment_id):
    """Run a single experiment in sequential mode (without Ray)."""
    return _run_experiment_impl(model_name, dataset_name, config, experiment_id)


def _run_experiment_impl(model_name, dataset_name, config, experiment_id):
    """Common implementation for both Ray and sequential experiments."""
    run_name = f"{model_name}_{dataset_name}"
    logger.info(f"Starting experiment: {run_name}")
    
    try:
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            
            if "dataset_config" in config and dataset_name in config["dataset_config"]:
                dataset_config = config["dataset_config"][dataset_name]
                logger.info(f"Using dataset-specific configuration for {dataset_name}")
                
                lookback = dataset_config.get("lookback", config["hyperparameters"].get("lookback", 96))
                horizon = dataset_config.get("horizon", config["hyperparameters"].get("horizon", 24))
                config_input_dim = dataset_config.get("input_dim", None)
                
                mlflow.log_param("lookback", lookback)
                mlflow.log_param("horizon", horizon)
                if config_input_dim:
                    mlflow.log_param("config_input_dim", config_input_dim)
            else:
                lookback = config["hyperparameters"].get("lookback", 96)
                horizon = config["hyperparameters"].get("horizon", 24)
                mlflow.log_param("lookback", lookback)
                mlflow.log_param("horizon", horizon)
            
            mlflow.log_param("model", model_name)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("epochs", config["hyperparameters"]["epochs"])
            mlflow.log_param("batch_size", config["hyperparameters"]["batch_size"])
            mlflow.log_param("lr", config["hyperparameters"]["lr"])
            
            device, device_info = select_device(config)
            mlflow.log_param("device", device.type)
            
            data = load_dataset(dataset_name, config)
            
            if data is not None:
                actual_input_dim = data.shape[1] if len(data.shape) > 1 else 1
            
            batch_size = config["hyperparameters"]["batch_size"]
            if "dataset_config" in config and dataset_name.lower() in config["dataset_config"]:
                if "batch_size" in config["dataset_config"][dataset_name.lower()]:
                    batch_size = config["dataset_config"][dataset_name.lower()]["batch_size"]
                    logger.info(f"Using dataset-specific batch size for {dataset_name}: {batch_size}")
            
            pin_memory = config["hyperparameters"].get("pin_memory", True)
            prefetch_factor = config["hyperparameters"].get("prefetch_factor", 2)
            persistent_workers = config["hyperparameters"].get("persistent_workers", True)
            
            train_loader, val_loader, test_loader = create_time_series_loaders(
                data,
                lookback=lookback,
                horizon=horizon,
                stride=config["hyperparameters"]["stride"],
                batch_size=batch_size,
                num_workers=config["hyperparameters"]["num_workers"],
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers
            )
            
            model_params = {
                "input_dim": actual_input_dim,
                "output_dim": actual_input_dim,
                "horizon": horizon,
                "dropout": config["hyperparameters"]["dropout"],
            }
            
            mlflow.log_param("actual_dimensions", f"{actual_input_dim}")
            
            if "InterferenceFANForecaster" in model_name:
                model_params.update({
                    "hidden_dim": config["hyperparameters"]["hidden_dim"],
                    "n_interference_layers": config["hyperparameters"]["n_layers"],
                })
            elif "FANForecaster" in model_name or "FANGatedForecaster" in model_name or "HybridPhaseFANForecaster" in model_name:
                model_params.update({
                    "hidden_dim": config["hyperparameters"]["hidden_dim"],
                    "n_fan_layers": config["hyperparameters"]["n_layers"],
                })
                
                if "HybridPhaseFANForecaster" in model_name:
                    model_params.update({
                        "phase_decay_epochs": config["hyperparameters"]["phase_decay_epochs"],
                    })
            elif "LSTMForecaster" in model_name:
                model_params.update({
                    "hidden_dim": config["hyperparameters"]["hidden_dim"],
                    "num_layers": config["hyperparameters"]["n_layers"],
                })
            elif "FANTransformerForecaster" in model_name or "PhaseOffsetTransformerForecaster" in model_name or "HybridPhaseFANTransformerForecaster" in model_name:
                use_checkpointing = config["hyperparameters"].get("use_checkpoint", False)
                
                model_params.update({
                    "d_model": config["hyperparameters"]["d_model"],
                    "fan_dim": config["hyperparameters"]["hidden_dim"],
                    "nhead": config["hyperparameters"]["n_heads"],
                    "num_layers": config["hyperparameters"]["n_layers"],
                    "use_checkpointing": use_checkpointing,
                })
                
                if "HybridPhaseFANTransformerForecaster" in model_name:
                    model_params.update({
                        "phase_decay_epochs": config["hyperparameters"]["phase_decay_epochs"],
                    })
            
            logger.info(f"Model parameters: {model_params}")
            
            for param_name, param_value in model_params.items():
                mlflow.log_param(param_name, param_value)
            
            model = create_model(model_name, model_params, device)
            
            logger.info(f"Model created: {model_name}")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
            
            mlflow.log_param("model_parameters", sum(p.numel() for p in model.parameters()))
            
            # Disable convergence tracking for now (missing ConvergenceTracker class)
            convergence_tracker = None
            
            history, best_model_state = train_model(
                model, train_loader, val_loader, config, device
            )
            
            # Extract losses from history
            train_losses = history["train_loss"]
            val_losses = history["val_loss"]
            
            metrics, _, _ = evaluate_model(model, test_loader, device)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            logger.info(f"Experiment completed: {run_name}, run_id: {run_id}")
            return run_id
            
    except Exception as e:
        logger.error(f"Experiment failed for {run_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


@ray.remote
def run_experiment(model_name, dataset_name, config, experiment_id):
    """Run a single experiment using Ray remote execution."""
    return _run_experiment_impl(model_name, dataset_name, config, experiment_id)


def save_run_ids(run_ids, experiment_name):
    if not run_ids:
        logger.warning("No run IDs to save")
        return

    time_series_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(time_series_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(results_dir, f"run_ids_{experiment_name}_{timestr}.json")

    with open(filename, "w") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "run_ids": run_ids,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(run_ids),
            },
            f,
            indent=2,
        )

    logger.info(f"Saved {len(run_ids)} run IDs to {filename}")
    return filename


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Time series forecasting experiments')
    parser.add_argument('--config', type=str, 
                        default=os.path.join(os.path.dirname(__file__), "config.yml"),
                        help='Path to config file')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning')
    args = parser.parse_args()
    
    setup_logger()
    
    # Load config first to check Ray settings
    from utils.config_utils import load_config
    config_path = args.config
    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Check Ray configuration
    ray_config = config.get("ray_config", {})
    fail_on_ray_error = ray_config.get("fail_on_ray_error", True)
    force_sequential = ray_config.get("force_sequential", False)
    
    if force_sequential:
        logger.info("Ray disabled by configuration (force_sequential=true)")
        ray_available = False
    else:
        ray_available = init_ray_for_mac()
        
        if not ray_available:
            if fail_on_ray_error:
                logger.error("Ray initialization failed on M2 MacBook Air!")
                logger.error("This will cause experiments to run sequentially and may produce inconsistent results.")
                logger.error("To fix this: 1) Check Ray installation, 2) Reduce num_cpus in init_ray_for_mac(), or 3) Set fail_on_ray_error=false in config.yml")
                raise RuntimeError("Ray initialization failed - aborting to prevent incorrect results")
            else:
                logger.warning("Ray initialization failed. Continuing with sequential mode (fail_on_ray_error=false).")
                logger.warning("CAUTION: Sequential mode may produce different results and will be significantly slower.")

    logger.info(f"Time series forecasting framework started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        models_to_run = config["models_to_run"]
        datasets_to_run = config["datasets_to_run"]
        
        total_experiments = len(models_to_run) * len(datasets_to_run)
        logger.info(f"Starting {total_experiments} experiments")
        
        experiment_name = config.get("experiment_name", "TimeSeriesBenchmark")
        
        import mlflow
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            except Exception as e:
                logger.error(f"Failed to create experiment: {e}")
                experiment_id = "0"
                logger.warning("Using default experiment (ID: 0)")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

        # Run experiments based on Ray availability
        if ray_available:
            logger.info("Running experiments in parallel using Ray")
            experiment_refs = []
            for model_name in models_to_run:
                for dataset_name in datasets_to_run:
                    logger.info(f"Scheduling experiment: model={model_name}, dataset={dataset_name}")
                    
                    ref = run_experiment.remote(model_name, dataset_name, config, experiment_id)
                    experiment_refs.append(ref)
            
            if experiment_refs:
                logger.info(f"Waiting for {len(experiment_refs)} experiments to complete...")
                try:
                    all_run_ids = []
                    for i, ref in enumerate(experiment_refs):
                        try:
                            run_id = ray.get(ref)
                            if run_id:
                                all_run_ids.append(run_id)
                            logger.info(f"Completed experiment {i+1}/{len(experiment_refs)}")
                        except Exception as e:
                            logger.error(f"Ray experiment {i+1} failed: {e}")
                            if fail_on_ray_error:
                                raise RuntimeError(f"Ray experiment execution failed: {e}")
                except Exception as e:
                    logger.error(f"Ray batch execution failed: {e}")
                    if fail_on_ray_error:
                        raise RuntimeError(f"Ray batch execution failed: {e}")
                    all_run_ids = []
                
                logger.info(f"All experiments completed, {len(all_run_ids)} successful runs")
            else:
                logger.warning("No experiments were scheduled")
                all_run_ids = []
        else:
            logger.info("Running experiments sequentially (Ray not available)")
            all_run_ids = []
            for model_name in models_to_run:
                for dataset_name in datasets_to_run:
                    logger.info(f"Running experiment: model={model_name}, dataset={dataset_name}")
                    
                    # Run experiment directly without Ray
                    run_id = run_experiment_sequential(model_name, dataset_name, config, experiment_id)
                    if run_id:
                        all_run_ids.append(run_id)
                    
            logger.info(f"All sequential experiments completed, {len(all_run_ids)} successful runs")

        logger.info("All experiments completed")

        if all_run_ids:
            save_run_ids(all_run_ids, experiment_name)
        else:
            logger.warning("No successful runs to save")
            
        logger.info("Experiments completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
