#!/usr/bin/env python
"""
Main entry point for the ML experimentation framework.
Runs experiments based on configuration settings.

Author: GitHub Copilot for keirparker
Last updated: 2025-02-26 17:26:16
"""

import mlflow
from loguru import logger
import os
import time
from datetime import datetime
import json
import numpy as np
import platform

from utils.config_utils import load_config, setup_environment
from utils.device_utils import select_device
from utils.data_utils import add_noise, make_sparse
from utils.training_utils import train_model, create_optimizer, create_scheduler
from utils.modelling_utils import evaluate_model
from utils.evaluation_utils import (
    generate_model_summary_table,
    plot_losses_by_epoch_comparison
)
from utils.visualisation_utils import (
    plot_training_history,
    plot_model_predictions,
    log_plots_to_mlflow,
    create_enhanced_visualizations
)
from src.signal_gen import get_periodic_data
from src.models import get_model_by_name


def setup_logger():
    """Configure the logger."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(
        f"{log_dir}/experiments_{time.strftime('%Y%m%d-%H%M%S')}.log",
        rotation="50 MB",  # Smaller log file size
        retention="5 days",  # Only keep logs for 5 days
        compression="zip",  # Compress log files
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(
        lambda msg: print(msg),
        level="INFO",
        format="{level} | <blue>{time:HH:mm:ss}</blue> | <level>{message}</level>",
    )


def run_experiment(model_name, dataset_type, data_version, config, experiment_id, experiment_name=None, run_number=None):
    """
    Runs a single experiment with the given configuration.

    Args:
        model_name: Name of the model to use
        dataset_type: Type of dataset to use
        data_version: Version of data to use (original, noisy, sparse)
        config: Configuration dictionary
        experiment_id: The MLflow experiment ID to log runs to
        experiment_name: Name of the experiment (for file organization)
        run_number: Run number identifier (for file organization)

    Returns:
        str: The MLflow run ID of the experiment
    """
    run_id = None
    try:
        # Create descriptive run name
        run_name = f"{model_name}_{dataset_type}_{data_version}"

        # We need to handle a potential issue where the experiment_id might be invalid
        # Strategy: Try with provided ID, fall back to default if needed
        try:
            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                logger.info(f"Started run {run_id} in experiment {experiment_id}")

                # Log experiment parameters
                mlflow.log_param("model", model_name)
                mlflow.log_param("dataset_type", dataset_type)
                mlflow.log_param("data_version", data_version)
                mlflow.log_param("num_samples", config["hyperparameters"]["num_samples"])
                mlflow.log_param("epochs", config["hyperparameters"]["epochs"])
                mlflow.log_param("lr", config["hyperparameters"]["lr"])

                # Add tags for easier filtering
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("dataset", dataset_type)
                mlflow.set_tag("data_version", data_version)
                mlflow.set_tag("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                mlflow.set_tag("user", os.environ.get("USER", "keirparker"))

                # Determine which device to use with enhanced detection
                device, device_info = select_device(config)
                mlflow.log_param("device", device.type)
                
                # Log detailed device information
                mlflow.log_param("device_name", device_info.get('name', 'Unknown'))
                
                if device_info.get('is_multi_gpu', False):
                    mlflow.log_param("gpu_count", device_info.get('gpu_count', 0))
                    mlflow.log_param("multi_gpu", True)
                
                # Log platform-specific optimizations
                if device_info.get('aws_instance', None):
                    mlflow.log_param("aws_instance", device_info['aws_instance'])
                
                if device_info.get('apple_silicon', False):
                    mlflow.log_param("apple_silicon", True)
                    if device_info.get('chip_model'):
                        mlflow.log_param("chip_model", device_info['chip_model'])
                        
                # Log Windows and Gigabyte GPU information if available
                if device_info.get('is_windows', False):
                    mlflow.log_param("windows", True)
                    mlflow.log_param("windows_version", platform.release())
                    
                    if device_info.get('is_windows_gigabyte', False):
                        mlflow.log_param("windows_gigabyte_gpu", True)
                        
                        # Log Gigabyte-specific GPU details if available
                        gigabyte_info = device_info.get('gigabyte_gpu_info', {})
                        if gigabyte_info:
                            if gigabyte_info.get('model'):
                                mlflow.log_param("gigabyte_gpu_model", gigabyte_info.get('model'))
                            if gigabyte_info.get('vram'):
                                mlflow.log_param("gigabyte_gpu_vram", f"{gigabyte_info.get('vram')} MB")
                            if gigabyte_info.get('driver_version'):
                                mlflow.log_param("nvidia_driver", gigabyte_info.get('driver_version'))

                # 1. Generate the base dataset + retrieve underlying function
                logger.info(f"Generating {dataset_type} dataset...")
                (t_train, data_train, t_test, data_test, data_config, true_func) = (
                    get_periodic_data(
                        periodic_type=dataset_type,
                        num_train_samples=config["hyperparameters"]["num_samples"],
                        num_test_samples=config["hyperparameters"]["test_samples"],
                    )
                )

                # Add the experiment batch name as a tag for grouping
                experiment_batch = config.get("experiment_name", "FAN_Model_Benchmark")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_id = f"{experiment_batch}_{timestamp}"
                mlflow.set_tag("batch_id", batch_id)
                mlflow.set_tag("experiment_batch", experiment_batch)

                # This second device selection is redundant and can be removed
                # The device has already been determined and logged above

                # 1. Generate the base dataset + retrieve underlying function (again)
                logger.info(f"Generating {dataset_type} dataset...")
                (t_train, data_train, t_test, data_test, data_config, true_func) = (
                    get_periodic_data(
                        periodic_type=dataset_type,
                        num_train_samples=config["hyperparameters"]["num_samples"],
                        num_test_samples=config["hyperparameters"]["test_samples"],
                    )
                )

                # Log dataset sizes
                mlflow.log_metric("train_size", len(t_train))
                mlflow.log_metric("test_size", len(t_test))

                # 2. Apply data transformation if needed
                if data_version == "original":
                    logger.info("Using original data (no transformation).")
                elif data_version == "noisy":
                    noise_level = config["hyperparameters"]["noise_level"]
                    logger.info(
                        f"Applying noise (level={noise_level}) to training & test data."
                    )
                    data_train = add_noise(data_train, noise_level=noise_level)
                    data_test = add_noise(data_test, noise_level=noise_level)
                    mlflow.log_param("noise_level", noise_level)
                elif data_version == "sparse":
                    sparsity_factor = config["hyperparameters"]["sparsity_factor"]
                    logger.info(f"Making data sparse (factor={sparsity_factor}).")
                    data_train, idx_train = make_sparse(
                        data_train, sparsity_factor=sparsity_factor
                    )
                    data_test, idx_test = make_sparse(
                        data_test, sparsity_factor=sparsity_factor
                    )
                    # Also shrink time arrays
                    t_train = t_train[idx_train]
                    t_test = t_test[idx_test]
                    mlflow.log_param("sparsity_factor", sparsity_factor)
                    # Log reduced sizes
                    mlflow.log_metric("train_size_after_sparse", len(t_train))
                    mlflow.log_metric("test_size_after_sparse", len(t_test))
                else:
                    logger.error(f"Unknown data version: {data_version}")
                    raise ValueError("Invalid data version")

                # Optional: Split training data for validation
                validation_split = config["hyperparameters"].get("validation_split", 0.1)
                mlflow.log_param("validation_split", validation_split)

                # 3. Retrieve & train the model
                logger.info(f"Initializing model: {model_name}")
                model = get_model_by_name(model_name)
                model.to(device)

                # Set up optimizer and scheduler early for checkpoint loading
                optimizer = create_optimizer(model, config)
                scheduler = create_scheduler(optimizer, config)
                
                # Check for existing checkpoint to resume training
                resume_training = config.get("resume_training", False)
                start_epoch = 0
                checkpoint_dir = os.path.join("models", "checkpoints")
                
                if resume_training:
                    # Look for best model checkpoint first, then most recent epoch
                    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                    
                    if os.path.exists(best_model_path):
                        # Try to resume from best model
                        checkpoint_path = best_model_path
                        logger.info(f"Found best model checkpoint, attempting to resume training")
                    else:
                        # Look for the latest epoch checkpoint
                        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
                        if checkpoint_files:
                            # Sort by epoch number
                            latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
                            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                            logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                        else:
                            checkpoint_path = None
                            logger.info("No checkpoints found, starting training from scratch")
                    
                    if checkpoint_path:
                        # Load the checkpoint
                        from utils.training_utils import load_checkpoint
                        start_epoch, loaded_history, loaded_config = load_checkpoint(
                            model, optimizer, scheduler, checkpoint_path
                        )
                        
                        if start_epoch > 0 and loaded_history:
                            logger.info(f"Resuming training from epoch {start_epoch}")
                            # Update config with any missing values from loaded config
                            if loaded_config:
                                for k, v in loaded_config.items():
                                    if k not in config:
                                        config[k] = v
                        else:
                            logger.warning("Failed to load checkpoint, starting from scratch")
                            start_epoch = 0

                # Track training start time for performance metrics
                train_start_time = time.time()

                # Train the model
                logger.info(f"Training model with validation_split={validation_split}")
                history = train_model(
                    model,
                    t_train,
                    data_train,
                    config,
                    device,
                    validation_split=validation_split,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    start_epoch=start_epoch
                )

                # Calculate and log training time
                training_time = time.time() - train_start_time
                mlflow.log_metric("training_time_seconds", training_time)
                mlflow.log_metric("last epoch", history["epochs"][-1])
                logger.info(f"Training completed in {training_time:.2f} seconds")

                # 4. Evaluate on test set
                logger.info("Evaluating model on test data")
                try:
                    test_metrics, predictions = evaluate_model(model, t_test, data_test, device)
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    # Provide a basic set of metrics to allow the run to continue
                    test_metrics = {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "r2": float('nan')}
                    predictions = np.zeros_like(data_test)  # Create dummy predictions

                # 5. Log metrics to MLflow - with expanded NaN handling
                try:
                    # Check for NaN values in metrics and replace them for logging
                    safe_metrics = {}
                    for metric_name, metric_value in test_metrics.items():
                        if isinstance(metric_value, (int, float)) and (np.isnan(metric_value) or np.isinf(metric_value)):
                            logger.warning(f"Found NaN/Inf in metric {metric_name}, replacing with fallback value")
                            if metric_name == "r2":
                                safe_metrics[metric_name] = -1.0  # Fallback for R2
                            else:
                                safe_metrics[metric_name] = 99.0  # Fallback for error metrics
                        else:
                            safe_metrics[metric_name] = metric_value
                    
                    # Log the sanitized metrics
                    mlflow.log_metric("test_mse", safe_metrics["mse"])
                    mlflow.log_metric("test_rmse", safe_metrics["rmse"])
                    mlflow.log_metric("test_mae", safe_metrics["mae"])
                    mlflow.log_metric("test_r2", safe_metrics["r2"])
                except Exception as e:
                    logger.error(f"Error logging metrics: {e}")
                    # Log fallback metrics to prevent run failure
                    mlflow.log_metric("test_mse", 999999)
                    mlflow.log_metric("test_rmse", 999999)
                    mlflow.log_metric("test_mae", 999999)
                    mlflow.log_metric("test_r2", -999999)
                
                # Log efficiency metrics if available
                if "flops" in test_metrics:
                    mlflow.log_metric("flops", test_metrics["flops"])
                    mlflow.log_metric("mflops", test_metrics["mflops"])
                    mlflow.log_metric("num_params", test_metrics["num_params"])
                    mlflow.log_metric("inference_time_ms", test_metrics["inference_time_ms"])

                # Log final and min training loss values (without every epoch)
                if history["train_loss"]:
                    mlflow.log_metric("final_train_loss", history["train_loss"][-1])
                    mlflow.log_metric("min_train_loss", min(history["train_loss"]))

                    # Only log a subset of epochs for the training curve (start, middle, end)
                    epochs = len(history["train_loss"])
                    if epochs <= 10:
                        # Log all epochs for short training runs
                        for i, loss in enumerate(history["train_loss"]):
                            mlflow.log_metric(f"train_loss_epoch_{i + 1}", loss)
                    else:
                        # For longer runs, log only key epochs: first, last, min, and a few samples
                        # First epoch
                        mlflow.log_metric("train_loss_epoch_1", history["train_loss"][0])
                        # Last epoch
                        mlflow.log_metric(
                            f"train_loss_epoch_{epochs}", history["train_loss"][-1]
                        )
                        # Epoch with minimum loss
                        min_loss_idx = history["train_loss"].index(min(history["train_loss"]))
                        mlflow.log_metric(
                            f"train_loss_epoch_{min_loss_idx + 1}",
                            history["train_loss"][min_loss_idx],
                        )
                        # A few samples throughout training (e.g., every 25% of training)
                        sample_points = [
                            int(epochs * 0.25),
                            int(epochs * 0.5),
                            int(epochs * 0.75),
                        ]
                        for point in sample_points:
                            if 1 < point < epochs:
                                mlflow.log_metric(
                                    f"train_loss_epoch_{point}",
                                    history["train_loss"][point - 1],
                                )

                # Handle validation loss similarly to training loss
                if history["val_loss"] and len(history["val_loss"]) > 0:
                    mlflow.log_metric("final_val_loss", history["val_loss"][-1])
                    mlflow.log_metric("min_val_loss", min(history["val_loss"]))

                    # Similar sampling approach for validation loss
                    epochs = len(history["val_loss"])
                    if epochs <= 10:
                        for i, loss in enumerate(history["val_loss"]):
                            mlflow.log_metric(f"val_loss_epoch_{i + 1}", loss)
                    else:
                        # First epoch
                        mlflow.log_metric("val_loss_epoch_1", history["val_loss"][0])
                        # Last epoch
                        mlflow.log_metric(
                            f"val_loss_epoch_{epochs}", history["val_loss"][-1]
                        )
                        # Epoch with minimum loss
                        min_loss_idx = history["val_loss"].index(min(history["val_loss"]))
                        mlflow.log_metric(
                            f"val_loss_epoch_{min_loss_idx + 1}",
                            history["val_loss"][min_loss_idx],
                        )
                        # A few samples
                        sample_points = [
                            int(epochs * 0.25),
                            int(epochs * 0.5),
                            int(epochs * 0.75),
                        ]
                        for point in sample_points:
                            if 1 < point < epochs:
                                mlflow.log_metric(
                                    f"val_loss_epoch_{point}",
                                    history["val_loss"][point - 1],
                                )

                # Log only key metrics from epoch history rather than every epoch
                if "metrics" in history and history["metrics"]:
                    if history["metrics"] and len(history["metrics"]) > 0:
                        for metric_name in history["metrics"][0].keys():
                            # Extract the values across epochs for this metric
                            metric_values = [m.get(metric_name, None) for m in history["metrics"]]
                            metric_values = [v for v in metric_values if v is not None]

                            if metric_values:
                                # Log final value
                                mlflow.log_metric(f"final_{metric_name}", metric_values[-1])

                                # For R², we want the maximum (best) value
                                if metric_name == "r2":
                                    best_value = max(metric_values)
                                    best_epoch = metric_values.index(best_value) + 1
                                    mlflow.log_metric(f"best_{metric_name}", best_value)
                                    mlflow.log_metric(f"best_{metric_name}_epoch", best_epoch)
                                # For error metrics, we want the minimum (best) value
                                else:
                                    best_value = min(metric_values)
                                    best_epoch = metric_values.index(best_value) + 1
                                    mlflow.log_metric(f"best_{metric_name}", best_value)
                                    mlflow.log_metric(f"best_{metric_name}_epoch", best_epoch)

                # 6. Generate and log plots
                # Training history plots
                history_plots = plot_training_history(
                    history, 
                    model_name, 
                    dataset_type, 
                    data_version,
                    experiment_name=experiment_name,
                    run_number=run_number
                )
                # Prediction plot
                prediction_plot = plot_model_predictions(
                    model_name,
                    dataset_type,
                    data_version,
                    t_train,
                    data_train,
                    t_test,
                    data_test,
                    predictions,
                    true_func,
                    experiment_name=experiment_name,
                    run_number=run_number
                )
                # Log all plots
                log_plots_to_mlflow(history_plots + [prediction_plot])

                # 7. Optionally save the model
                if config.get("save_model", False):
                    model_dir = os.path.join("models", f"{model_name}_{dataset_type}_{data_version}")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, "model.pt")
                    # torch.save(model.state_dict(), model_path)
                    mlflow.log_artifact(model_path)

                # Add summary for this run
                mlflow.set_tag("status", "COMPLETED")
                mlflow.set_tag("execution_time", f"{training_time:.2f}s")

                logger.info(
                    f"Completed experiment: model={model_name}, dataset={dataset_type}, version={data_version}. "
                    f"Test metrics: R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}"
                )

                return run_id

        except Exception as e:
            logger.error(f"Error during experiment: {e}")
            if mlflow.active_run():
                mlflow.set_tag("status", "FAILED")
                mlflow.set_tag("error_message", str(e))
            raise

        return run_id

    except Exception as outer_e:
        logger.error(f"Outer error in run_experiment: {outer_e}")
        raise


def save_run_ids(run_ids, experiment_name, run_number=None):
    """
    Save run IDs to a JSON file for later analysis.

    Args:
        run_ids: List of run IDs
        experiment_name: Name of the experiment
        run_number: Run number (optional)
    """
    if not run_ids:
        logger.warning("No run IDs to save")
        return

    # Clean experiment name for safe path usage
    safe_exp_name = experiment_name.replace(" ", "_").replace("/", "_")
    
    # Determine directory and filename
    if run_number is not None:
        # Create experiment directory structure
        exp_dir = f"results/{safe_exp_name}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create run directory
        run_dir = f"{exp_dir}/run_{run_number}"
        os.makedirs(run_dir, exist_ok=True)
        
        # Set filename without timestamp since we have run number
        filename = f"{run_dir}/run_ids.json"
    else:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Generate filename with timestamp for backward compatibility
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"results/run_ids_{safe_exp_name}_{timestr}.json"

    # Save to file
    with open(filename, "w") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "run_ids": run_ids,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": os.environ.get("USER", "keirparker"),
                "count": len(run_ids),
                "run_number": run_number,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved {len(run_ids)} run IDs to {filename}")

def test_gpu_functionality():
    """
    Run a quick test to verify if the GPU is properly detected and functioning.
    """
    logger.info("=== GPU/CUDA DETECTION TEST ===")
    import torch
    
    # Basic CUDA availability check
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Device count: {torch.cuda.device_count()}")
        
        # Get device name and properties
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Device name: {device_name}")
        
        # Try to get memory info
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            logger.info(f"VRAM: {total_mem / (1024**3):.2f} GB total, {free_mem / (1024**3):.2f} GB free")
        except:
            logger.info("Could not query GPU memory info")
        
        # Test tensor operations
        try:
            logger.info("Running GPU test tensor operation...")
            # Use a reasonably sized tensor to verify computation without using too much memory
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            
            # Record start time
            start_time = time.time()
            
            # Perform matrix multiplication (computationally intensive operation)
            z = torch.matmul(x, y)
            
            # Record end time and calculate duration
            duration = time.time() - start_time
            
            # Force synchronization to ensure operation is complete
            torch.cuda.synchronize()
            
            logger.info(f"GPU test successful! Result shape: {z.shape}")
            logger.info(f"Operation completed in {duration*1000:.2f} ms")
            
            # Clean up GPU memory
            del x, y, z
            torch.cuda.empty_cache()
            
            logger.info("GPU test passed ✓")
        except Exception as e:
            logger.error(f"GPU test failed: {e}")
    else:
        logger.warning("CUDA is not available. Will run on CPU instead.")
    
    logger.info("================================")

def main():
    """
    Main entry point. Loads configuration and runs all specified experiments.
    """
    # Setup logging
    setup_logger()
    
    # Run GPU test at startup
    test_gpu_functionality()

    # Clean up disk space before starting
    mlflow_dir = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
    if os.path.exists(mlflow_dir):
        try:
            experiment_dirs = [d for d in os.listdir(mlflow_dir) 
                              if d.isdigit() or (d.replace("-", "").isdigit() and d != "0")]
            if len(experiment_dirs) > 3:
                # Sort by creation time (oldest first)
                experiment_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(mlflow_dir, x)))
                # Remove all but the 3 newest experiment directories
                for old_dir in experiment_dirs[:-3]:
                    old_path = os.path.join(mlflow_dir, old_dir)
                    logger.info(f"Cleaning up old MLflow data: {old_path}")
                    import shutil
                    shutil.rmtree(old_path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error cleaning up MLflow directories: {e}")
    
    # Also clean old log files
    log_dir = "logs"
    if os.path.exists(log_dir):
        try:
            log_files = [f for f in os.listdir(log_dir) if f.endswith(".log") and not f.endswith(".zip")]
            if len(log_files) > 10:  # Keep only 10 most recent uncompressed log files
                log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
                for old_log in log_files[:-10]:
                    old_log_path = os.path.join(log_dir, old_log)
                    logger.info(f"Removing old log file: {old_log_path}")
                    os.remove(old_log_path)
        except Exception as e:
            logger.warning(f"Error cleaning up log files: {e}")

    # Log script start
    logger.info(
        f"ML experiment framework started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Running as user: {os.environ.get('USER', 'keirparker')}")

    try:
        # Load and set up the environment
        config = load_config()
        setup_environment(config)

        # Get experiment parameters
        models_to_run = config["models_to_run"]
        dataset_types = config["datasets_to_run"]
        data_versions = config["data_versions"]

        total_experiments = len(models_to_run) * len(dataset_types) * len(data_versions)
        logger.info(f"Starting {total_experiments} experiments")

        # Define experiment
        experiment_name = config.get("experiment_name", "FAN_Model_Benchmark")
        
        # Get run number if provided in config, otherwise generate a sequential one
        run_number = config.get("run_number")
        if run_number is None:
            # Try to determine the next run number based on existing directories
            safe_exp_name = experiment_name.replace(" ", "_").replace("/", "_")
            exp_dir = f"results/{safe_exp_name}"
            if os.path.exists(exp_dir):
                existing_runs = [d for d in os.listdir(exp_dir) if d.startswith("run_")]
                if existing_runs:
                    # Extract numbers from run_X directories
                    existing_numbers = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
                    run_number = max(existing_numbers) + 1 if existing_numbers else 1
                else:
                    run_number = 1
            else:
                run_number = 1
            
            logger.info(f"Automatically assigned run number: {run_number}")

        # Get or create the experiment
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(
                    f"Created new experiment: {experiment_name} (ID: {experiment_id})"
                )
            except Exception as e:
                logger.error(f"Failed to create experiment: {e}")
                # If we can't create it, use the default experiment
                experiment_id = "0"  # Default experiment ID in MLflow
                logger.warning("Using default experiment (ID: 0)")
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing experiment: {experiment_name} (ID: {experiment_id})"
            )

        # Store all run IDs for comparison
        all_run_ids = []

        # Run all combinations
        for model_name in models_to_run:
            for dataset_type in dataset_types:
                for data_version in data_versions:
                    logger.info(
                        f"Running experiment: model={model_name}, dataset={dataset_type}, version={data_version}"
                    )
                    try:
                        run_id = run_experiment(
                            model_name,
                            dataset_type,
                            data_version,
                            config,
                            experiment_id,
                            experiment_name=experiment_name,
                            run_number=run_number
                        )
                        all_run_ids.append(run_id)
                    except Exception as e:
                        logger.error(f"Error in experiment: {e}")
                        import traceback

                        logger.error(traceback.format_exc())

        logger.info("All experiments completed")

        # Save run IDs for later analysis
        save_run_ids(all_run_ids, experiment_name, run_number)

        # Generate summary table - make sure it goes to the same experiment
        if all_run_ids:
            logger.info("Generating summary table...")
            # Pass explicit experiment_id to ensure proper MLflow logging
            summary_df = generate_model_summary_table(
                all_run_ids, 
                experiment_name, 
                run_number=run_number
            )
            if summary_df is not None:
                logger.info(f"Summary table generated with {len(summary_df)} runs")
            else:
                logger.warning("Failed to generate summary table")

            # NEW CODE: Generate loss comparison plots
            logger.info("Generating loss comparison plots...")

            # Training loss comparison with no smoothing and markers visible
            train_loss_plot = plot_losses_by_epoch_comparison(
                run_ids=all_run_ids, 
                metric_name="train_loss", 
                include_validation=True,  # Show validation loss too
                smooth_factor=None  # No smoothing - show raw data
            )

            # # Combined training and validation loss comparison
            # combined_loss_plot = plot_losses_by_epoch_comparison(
            #     run_ids=all_run_ids,
            #     metric_name="train_loss",
            #     include_validation=False,
            #     smooth_factor=3,  # Apply some smoothing for better readability
            # )

            # Log the plots to MLflow
            if train_loss_plot:
                if all_run_ids:
                    logger.info("Generating enhanced visualizations...")
                    create_enhanced_visualizations(
                        all_run_ids, experiment_id, experiment_name, run_number
                    )




    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        # Make sure no MLflow runs are left open
        if mlflow.active_run():
            mlflow.end_run()

        # Log script end
        logger.info(
            f"ML experiment framework finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )



if __name__ == "__main__":
    main()
