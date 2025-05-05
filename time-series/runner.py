#!/usr/bin/env python
"""
Time Series Forecasting Runner

This script runs time series forecasting experiments with various FAN-based models.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import mlflow
from loguru import logger
import time
from datetime import datetime
import yaml
import json

from time_series.data import create_time_series_loaders, TimeSeriesDataset
from time_series.models import get_model_by_name, list_available_models

from time_series.utils.config_utils import setup_environment
from time_series.utils.device_utils import select_device
from time_series.utils.training_utils import create_optimizer, create_scheduler
from time_series.utils.evaluation_utils import plot_losses_by_epoch_comparison
from time_series.utils.convergence_utils import calculate_convergence_speed, calculate_training_efficiency, compare_convergence


def setup_logger():
    """Configure the logger."""
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
    """Load YAML configuration file."""
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


def load_dataset(dataset_name, config):
    """Load and prepare a time series dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # ETT datasets
    if dataset_name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        from time_series.data.ts_dataset import load_ett_dataset
        data = load_ett_dataset(dataset_name)
        logger.info(f"Loaded {dataset_name} with shape {data.shape}")
    
    # Multivariate datasets from https://github.com/laiguokun/multivariate-time-series-data
    elif dataset_name.lower() in ["electricity", "traffic", "solar-energy"]:
        from time_series.data.ts_dataset import load_multivariate_dataset
        
        norm_method = "standard"
        if "dataset_config" in config and dataset_name.lower() in config["dataset_config"]:
            norm_method = config["dataset_config"][dataset_name.lower()].get("norm_method", "standard")
            
        normalize = True
        if norm_method.lower() == "none":
            normalize = False
            
        actual_name = dataset_name.lower()
        
        memory_efficient = True
        if "memory_efficient" in config.get("dataset_config", {}).get(actual_name, {}):
            memory_efficient = config["dataset_config"][actual_name]["memory_efficient"]
        
        data = load_multivariate_dataset(
            actual_name, 
            normalize=normalize,
            memory_efficient=memory_efficient
        )
        
        logger.info(f"Loaded {dataset_name} with shape {data.shape} (normalization: {norm_method}, memory-efficient: {memory_efficient}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: ETTh1, ETTh2, ETTm1, ETTm2, electricity, traffic, solar-energy")
    
    return data


def train_model(model, train_loader, val_loader, config, device):
    """Train a time series forecasting model."""
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
        logger.info(f"Batch logging every {batch_log_interval} batches")
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
    """Evaluate model on test set."""
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


def run_experiment(model_name, dataset_name, config, experiment_id):
    """Run a single time series forecasting experiment."""
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
            
            if "FANForecaster" in model_name or "FANGatedForecaster" in model_name or "HybridPhaseFANForecaster" in model_name:
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
            elif "TransformerForecaster" in model_name:
                model_params.update({
                    "hidden_dim": config["hyperparameters"]["d_model"],
                    "nhead": config["hyperparameters"]["n_heads"],
                    "num_layers": config["hyperparameters"]["n_layers"],
                })
                
            try:
                logger.info(f"Creating model '{model_name}' with params: {model_params}")
                
                model = get_model_by_name(model_name, **model_params)
            except ValueError as e:
                available_models = list_available_models()
                logger.error(f"Failed to create model {model_name}. Available models: {available_models}")
                raise ValueError(f"Unknown model: {model_name} not found in model registry: {str(e)}")
                
            mlflow.log_param("num_params", sum(p.numel() for p in model.parameters()))
                
            logger.info(f"Training {model_name} on {dataset_name}")
            train_start_time = time.time()
            history, best_model_state = train_model(model, train_loader, val_loader, config, device)
            training_time = time.time() - train_start_time
            
            mlflow.log_metric("training_time", training_time)
            mlflow.log_metric("final_train_loss", history["train_loss"][-1])
            mlflow.log_metric("final_val_loss", history["val_loss"][-1])
            mlflow.log_metric("best_val_loss", min(history["val_loss"]))
            
            logger.info(f"Calculating convergence metrics for {model_name}")
            convergence_threshold = config.get("convergence_threshold", 0.1)
            
            convergence_epoch = calculate_convergence_speed(
                history, target_metric='val_loss', threshold=convergence_threshold)
            
            if convergence_epoch is not None:
                mlflow.log_metric("convergence_epoch", convergence_epoch)
                mlflow.log_metric("convergence_speed", 1.0 / convergence_epoch if convergence_epoch > 0 else 0)
                
                if "epoch_times" in history and len(history["epoch_times"]) >= convergence_epoch:
                    time_to_convergence = sum(history["epoch_times"][:convergence_epoch])
                    mlflow.log_metric("time_to_convergence", time_to_convergence)
                elif "training_time" in history and len(history["training_time"]) >= convergence_epoch:
                    time_to_convergence = sum(history["training_time"][:convergence_epoch])
                    mlflow.log_metric("time_to_convergence", time_to_convergence)
                    
                for threshold_pct in [0.1, 0.05, 0.01]:
                    convergence_epoch_pct = calculate_convergence_speed(
                        history, target_metric='val_loss', threshold=threshold_pct)
                    if convergence_epoch_pct is not None:
                        mlflow.log_metric(f"convergence_epoch_{int(100*(1-threshold_pct))}", convergence_epoch_pct)
                        
            model_has_phase_offset = "PhaseOffset" in model_name
            mlflow.log_param("has_phase_offset", model_has_phase_offset)
            
            logger.info(f"Evaluating {model_name} on {dataset_name} test set")
            test_metrics, predictions, targets = evaluate_model(model, test_loader, device)
            
            for name, value in test_metrics.items():
                mlflow.log_metric(f"test_{name}", value)
                
            for i, (train_loss, val_loss) in enumerate(zip(history["train_loss"], history["val_loss"])):
                epoch = i + 1
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                
            if config.get("save_best", False):
                min_improvement = config.get("min_improvement", 0.005)
                
                if best_model_state is not None:
                    checkpoint_dir = "checkpoints"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
                    
                    logger.info(f"Saving single best model checkpoint for {model_name} with val_loss={min(history['val_loss']):.6f}")
                    
                    torch.save({
                        "model_state_dict": best_model_state,
                        "config": config,
                        "history": history,
                        "val_loss": min(history["val_loss"])
                    }, checkpoint_path)
                    mlflow.log_artifact(checkpoint_path)
                else:
                    logger.warning("No best model state to save (no significant improvements found)")
            
            logger.info(f"Experiment completed: {run_name}")
            return run_id
            
    except Exception as e:
        logger.error(f"Error in experiment {run_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def save_run_ids(run_ids, experiment_name):
    """Save run IDs to a JSON file for later analysis."""
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


def generate_performance_table(run_ids, experiment_name):
    """Generate a academic-style performance comparison table for time series forecasting models."""
    import os
    import numpy as np
    from collections import defaultdict
    
    logger.info(f"Generating performance table from {len(run_ids)} runs")
    
    client = mlflow.tracking.MlflowClient()
    
    results = defaultdict(lambda: defaultdict(dict))
    
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            
            model = run.data.params.get("model", "Unknown")
            dataset = run.data.params.get("dataset", "Unknown")
            
            if "test_mse" not in run.data.metrics or "test_mae" not in run.data.metrics:
                logger.warning(f"Skipping incomplete run: {model}_{dataset} (Run ID: {run_id})")
                continue
                
            results[f"{model}_{dataset}"]["test_mse"] = run.data.metrics.get("test_mse", float('nan'))
            results[f"{model}_{dataset}"]["test_mae"] = run.data.metrics.get("test_mae", float('nan'))
            results[f"{model}_{dataset}"]["test_rmse"] = run.data.metrics.get("test_rmse", float('nan'))
            
            results[f"{model}_{dataset}"]["model"] = model
            results[f"{model}_{dataset}"]["dataset"] = dataset
            results[f"{model}_{dataset}"]["input_dim"] = run.data.params.get("actual_dimensions", "?")
            results[f"{model}_{dataset}"]["lookback"] = run.data.params.get("lookback", "?")
            results[f"{model}_{dataset}"]["horizon"] = run.data.params.get("horizon", "?")
            
        except Exception as e:
            logger.error(f"Error processing run {run_id}: {e}")
    
    if not results:
        logger.warning("No valid results found to generate table")
        return None
    
    df = pd.DataFrame.from_dict(results, orient='index')
    
    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())
    
    model_display_names = {
        "TransformerForecaster": "Transformer",
        "FANTransformerForecaster": "FAN-T",
        "PhaseOffsetTransformerForecaster": "PO-T",
        "FANForecaster": "FAN",
        "FANGatedForecaster": "FAN-G"
    }
    
    dataset_display_names = {
        "electricity": "Electricity",
        "traffic": "Traffic",
        "solar-energy": "Solar"
    }
    
    markdown_output = f"# FAN-based Transformer Model Performance\n\n"
    
    markdown_output += "## Test Set Performance Metrics\n\n"
    
    markdown_output += "| Model | "
    
    for dataset in datasets:
        display_name = dataset_display_names.get(dataset, dataset.capitalize())
        markdown_output += f"{display_name}-MSE | {display_name}-MAE | "
    
    markdown_output += "Avg. MSE | Avg. MAE | Improvement |\n"
    
    markdown_output += "|-------|"
    for _ in datasets:
        markdown_output += "-------|-------|"
    markdown_output += "-------|-------|-------|\n"
    
    baseline_model = "TransformerForecaster"
    if baseline_model not in models and models:
        baseline_model = models[0]
    
    model_averages = {}
    for model in models:
        mse_values = []
        mae_values = []
        
        for dataset in datasets:
            model_dataset_key = f"{model}_{dataset}"
            if model_dataset_key in results:
                mse = results[model_dataset_key]["test_mse"]
                mae = results[model_dataset_key]["test_mae"]
                mse_values.append(mse)
                mae_values.append(mae)
                
        if mse_values and mae_values:
            avg_mse = np.mean(mse_values)
            avg_mae = np.mean(mae_values)
            model_averages[model] = (avg_mse, avg_mae)
    
    if baseline_model in model_averages:
        baseline_mse, baseline_mae = model_averages[baseline_model]
    else:
        baseline_mse, baseline_mae = float('nan'), float('nan')
    
    best_mse_by_dataset = {}
    best_mae_by_dataset = {}
    
    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset]
        if not dataset_df.empty:
            best_mse_by_dataset[dataset] = dataset_df["test_mse"].min()
            best_mae_by_dataset[dataset] = dataset_df["test_mae"].min()
    
    if model_averages:
        best_avg_mse = min(model_averages.values(), key=lambda x: x[0])[0]
        best_avg_mae = min(model_averages.values(), key=lambda x: x[1])[1]
    else:
        best_avg_mse = float('nan')
        best_avg_mae = float('nan')
    
    sorted_models = sorted(
        [m for m in models if m in model_averages],
        key=lambda x: model_averages[x][0]
    )
    
    for model in sorted_models:
        display_name = model_display_names.get(model, model)
        markdown_output += f"| {display_name} | "
        
        for dataset in datasets:
            model_dataset_key = f"{model}_{dataset}"
            
            if model_dataset_key in results:
                mse = results[model_dataset_key]["test_mse"]
                mae = results[model_dataset_key]["test_mae"]
                
                mse_str = f"**{mse:.4f}**" if abs(mse - best_mse_by_dataset[dataset]) < 1e-6 else f"{mse:.4f}"
                mae_str = f"**{mae:.4f}**" if abs(mae - best_mae_by_dataset[dataset]) < 1e-6 else f"{mae:.4f}"
                
                markdown_output += f"{mse_str} | {mae_str} | "
            else:
                markdown_output += "- | - | "
        
        avg_mse, avg_mae = model_averages[model]
        
        avg_mse_str = f"**{avg_mse:.4f}**" if abs(avg_mse - best_avg_mse) < 1e-6 else f"{avg_mse:.4f}"
        avg_mae_str = f"**{avg_mae:.4f}**" if abs(avg_mae - best_avg_mae) < 1e-6 else f"{avg_mae:.4f}"
        
        if model == baseline_model:
            imp_str = "-"
        else:
            if not np.isnan(baseline_mse) and baseline_mse != 0:
                mse_improvement = (baseline_mse - avg_mse) / baseline_mse * 100
                
                imp_str = f"**{mse_improvement:.1f}%**" if abs(avg_mse - best_avg_mse) < 1e-6 else f"{mse_improvement:.1f}%"
            else:
                imp_str = "-"
        
        markdown_output += f"{avg_mse_str} | {avg_mae_str} | {imp_str} |\n"
    
    if not df.empty:
        lookback = df["lookback"].iloc[0] if len(df["lookback"].unique()) == 1 else "varied"
        horizon = df["horizon"].iloc[0] if len(df["horizon"].unique()) == 1 else "varied"
    else:
        lookback = "unknown"
        horizon = "unknown"
    
    markdown_output += f"\n**Experimental Settings:** Lookback={lookback}, Horizon={horizon}, Experiment=\"{experiment_name}\"\n\n"
    markdown_output += f"**Note:** Best values are in **bold**. Improvement percentages are relative to the Transformer baseline."
    
    time_series_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(time_series_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(results_dir, f"results_{experiment_name}_{timestr}.md")
    
    with open(output_path, 'w') as f:
        f.write(markdown_output)
    
    logger.info(f"Performance table saved to {output_path}")
    
    try:
        csv_data = []
        
        short_names = {model: model_display_names.get(model, model) for model in models}
        
        for model in sorted_models:
            row_data = {"Model": short_names[model]}
            
            for dataset in datasets:
                display_name = dataset_display_names.get(dataset, dataset.capitalize())
                model_dataset_key = f"{model}_{dataset}"
                
                if model_dataset_key in results:
                    row_data[f"{display_name}_MSE"] = results[model_dataset_key]["test_mse"]
                    row_data[f"{display_name}_MAE"] = results[model_dataset_key]["test_mae"]
                else:
                    row_data[f"{display_name}_MSE"] = np.nan
                    row_data[f"{display_name}_MAE"] = np.nan
            
            if model in model_averages:
                avg_mse, avg_mae = model_averages[model]
                row_data["Avg_MSE"] = avg_mse
                row_data["Avg_MAE"] = avg_mae
                
                if model != baseline_model and not np.isnan(baseline_mse) and baseline_mse != 0:
                    row_data["Improvement"] = (baseline_mse - avg_mse) / baseline_mse * 100
                else:
                    row_data["Improvement"] = np.nan
                    
                if isinstance(lookback, str):
                    row_data["Lookback"] = lookback
                else:
                    row_data["Lookback"] = str(lookback)
                
                if isinstance(horizon, str):
                    row_data["Horizon"] = horizon
                else:
                    row_data["Horizon"] = str(horizon)
                    
                row_data["Experiment"] = experiment_name
                
                csv_data.append(row_data)
        
        csv_df = pd.DataFrame(csv_data)
        
        csv_path = output_path.replace(".md", ".csv")
        csv_df.to_csv(csv_path, index=False, float_format="%.4f")
        
        tsv_path = output_path.replace(".md", ".tsv")
        csv_df.to_csv(tsv_path, index=False, sep='\t', float_format="%.4f")
        
        logger.info(f"CSV results saved to {csv_path}")
        logger.info(f"TSV results saved for academic papers to {tsv_path}")
        
    except Exception as e:
        logger.error(f"Error generating CSV/TSV data: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return output_path
    

def main():
    """Main entry point for time series forecasting experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Time series forecasting experiments')
    parser.add_argument('--config', type=str, 
                        default=os.path.join(os.path.dirname(__file__), "config.yml"),
                        help='Path to config file')
    args = parser.parse_args()
    
    setup_logger()
    
    logger.info(f"Time series forecasting framework started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        config_path = args.config
        logger.info(f"Loading config from: {config_path}")
        config = load_config(config_path)
        
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

        all_run_ids = []

        for model_name in models_to_run:
            for dataset_name in datasets_to_run:
                logger.info(f"Running experiment: model={model_name}, dataset={dataset_name}")
                try:
                    run_id = run_experiment(model_name, dataset_name, config, experiment_id)
                    if run_id:
                        all_run_ids.append(run_id)
                except Exception as e:
                    logger.error(f"Error in experiment: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        logger.info("All experiments completed")

        save_run_ids(all_run_ids, experiment_name)
        
        if all_run_ids:
            logger.info("Generating loss comparison plots...")
            
            train_loss_plot = plot_losses_by_epoch_comparison(
                run_ids=all_run_ids, 
                metric_name="train_loss", 
                include_validation=True
            )
            
            logger.info("Generating test metrics comparison...")
            test_metrics = ["test_mse", "test_rmse", "test_mae"]
            for metric in test_metrics:
                plot_losses_by_epoch_comparison(
                    run_ids=all_run_ids,
                    metric_name=metric,
                    include_validation=False
                )
                
            logger.info("Generating performance summary table...")
            generate_performance_table(
                run_ids=all_run_ids,
                experiment_name=experiment_name
            )
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if mlflow.active_run():
            mlflow.end_run()

        logger.info(f"Time series forecasting framework finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()