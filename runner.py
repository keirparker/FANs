#!/usr/bin/env python
"""
Main entry point for the Fourier Analysis Networks (FANs) experimentation framework.

This script provides a structured pipeline for running machine learning experiments with 
various models, datasets, and data transformations. It handles the full lifecycle of experiments:
- Loading and preparing data (original, noisy, sparse versions)
- Initializing and training models
- Evaluating performance
- Generating visualizations 
- Tracking experiments with MLflow
- Creating summary analyses and comparisons

Features:
- Support for both synthetic test functions and time series forecasting
- Multiple model types: FAN, FANGated, FANPhaseOffset variants
- Data transformation options: original, noisy, sparse
- Phase offset tracking and visualization for relevant models
- Comprehensive visualization and comparison tools

Author: GitHub Copilot for keirparker
Last updated: 2025-03-06
"""

import mlflow
from loguru import logger
import os
import time
from datetime import datetime
import json
import numpy as np
import glob

# Core utilities
from utils.config_utils import load_config, setup_environment
from utils.device_utils import select_device
from utils.data_utils import add_noise, make_sparse
from utils.training_utils import train_model
from utils.modelling_utils import evaluate_model

# Visualization and evaluation utilities
from utils.evaluation_utils import (
    generate_model_summary_table,
    plot_losses_by_epoch_comparison
)
from utils.visualisation_utils import (
    plot_training_history,
    plot_model_predictions,
    log_plots_to_mlflow,
    create_enhanced_visualizations,
    organize_plots_by_dataset_and_type,
    generate_data_type_comparison_table,
    plot_offset_evolution,
    plot_offset_convergence
)
from utils.efficiency_utils import (
    create_all_efficiency_visualizations
)

# Data generation
from src.signal_gen import get_periodic_data
from src.ts_data import get_etth1_data

# Models
from src.models import get_model_by_name

# Forecaster models


def setup_logger(log_level="INFO"):
    """
    Configure the logger with both file and console outputs.
    
    Sets up:
    - A file logger that records detailed logs with timestamps
    - A console logger with color-coded output for better readability
    
    Args:
        log_level: The logging level to use (e.g., "DEBUG", "INFO", "WARNING")
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Clear any existing handlers
    logger.remove()
    
    # Add file handler - always use DEBUG level for file logs to capture everything
    log_file = f"{log_dir}/experiments_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(
        log_file,
        rotation="500 MB",
        level="DEBUG",  # Always log debug info to file
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    
    # Add console handler with color formatting and requested level
    logger.add(
        lambda msg: print(msg),
        level=log_level,
        format="{level} | <blue>{time:HH:mm:ss}</blue> | <level>{message}</level>",
    )
    
    logger.info(f"Logger initialized, writing to {log_file} (console level: {log_level})")


def generate_visualization_plots(history, model_name, dataset_type, data_version, 
                           t_train, data_train, t_test, data_test, predictions, true_func, config):
    """
    Generate all visualization plots for a model run.
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        dataset_type: Type of dataset used
        data_version: Version of data used
        t_train: Training time points
        data_train: Training data
        t_test: Test time points
        data_test: Test data
        predictions: Model predictions on test data
        true_func: True function (if available)
        config: Configuration dictionary
        
    Returns:
        list: Paths to all generated plots
    """
    all_plots = []
    
    # Generate training history plots
    logger.info(f"Generating training history plots for {model_name}")
    history_plots = plot_training_history(history, model_name, dataset_type, data_version)
    all_plots.extend(history_plots)
    
    # Generate prediction plots
    logger.info(f"Generating prediction plots for {model_name}")
    prediction_plot = plot_model_predictions(
        model_name,
        dataset_type,
        data_version,
        t_train,
        data_train,
        t_test,
        data_test,
        predictions,
        true_func
    )
    all_plots.append(prediction_plot)
    
    # Generate phase offset plots if applicable
    offset_plots = []
    plot_offset_enabled = config.get("visualization", {}).get("offset_loss", False)
    plot_convergence = config.get("visualization", {}).get("plot_offset_convergence", False)
    
    if plot_offset_enabled and "PhaseOffset" in model_name and "offset_history" in history:
        logger.info(f"Generating phase offset evolution plots for {model_name}")
        
        # Get visualization config parameters
        viz_config = config.get("visualization", {})
        
        # Generate individual parameter plots
        offset_plots = plot_offset_evolution(
            history, 
            model_name, 
            dataset_type, 
            data_version, 
            zoom_y_axis=viz_config.get("zoom_y_axis", True),
            add_reference_lines=viz_config.get("add_reference_lines", False)
        )
        
        # Generate convergence plot if enabled
        if plot_convergence:
            convergence_plot = plot_offset_convergence(
                history, 
                model_name, 
                dataset_type, 
                data_version
            )
            if convergence_plot:
                offset_plots.append(convergence_plot)
                logger.info(f"Generated convergence plot for {model_name}")
        
        # Log phase offset metrics
        if offset_plots and "offset_history" in history:
            for name, values_list in history["offset_history"].items():
                if values_list:
                    # Get final offset values
                    final_offsets = values_list[-1]
                    # Log final mean offset
                    final_mean = float(np.mean(final_offsets))
                    mlflow.log_metric(f"final_mean_offset_{name}", final_mean)
                    
                    # Log min/max for multi-dimensional offsets
                    if hasattr(final_offsets, "__len__") and len(final_offsets) > 1:
                        mlflow.log_metric(f"final_min_offset_{name}", float(np.min(final_offsets)))
                        mlflow.log_metric(f"final_max_offset_{name}", float(np.max(final_offsets)))
                    
                    # Track rate of change during training
                    if len(values_list) > 1:
                        first_mean = float(np.mean(values_list[0]))
                        avg_change_per_epoch = (final_mean - first_mean) / len(values_list)
                        mlflow.log_metric(f"offset_avg_change_per_epoch_{name}", avg_change_per_epoch)
        
        all_plots.extend(offset_plots)
        logger.info(f"Generated {len(offset_plots)} phase offset plots total")
    
    # Log all plots to MLflow
    logger.info(f"Logging {len(all_plots)} plots to MLflow")
    log_plots_to_mlflow(all_plots)
    
    return all_plots


def log_metrics_and_history(test_metrics, history, training_time=None):
    """
    Log model metrics and training history to MLflow.
    
    Args:
        test_metrics: Dictionary of test metrics (MSE, RMSE, MAE, etc.)
        history: Dictionary with training history data
        training_time: Total training time in seconds (optional, already logged in run_experiment)
    """
    # Log test metrics
    mlflow.log_metric("test_mse", test_metrics["mse"])
    mlflow.log_metric("test_rmse", test_metrics["rmse"]) 
    mlflow.log_metric("test_mae", test_metrics["mae"])
    mlflow.log_metric("test_r2", test_metrics["r2"])
    mlflow.log_metric("test_mape", test_metrics["mape"])
    
    # Log train loss metrics
    if history["train_loss"]:
        mlflow.log_metric("final_train_loss", history["train_loss"][-1])
        mlflow.log_metric("min_train_loss", min(history["train_loss"]))
        
        # Log strategic epochs for training loss
        epochs = len(history["train_loss"])
        if epochs <= 10:
            # For short runs, log all epochs
            for i, loss in enumerate(history["train_loss"]):
                mlflow.log_metric(f"train_loss_epoch_{i + 1}", loss)
        else:
            # For longer runs, log key points
            # First epoch
            mlflow.log_metric("train_loss_epoch_1", history["train_loss"][0])
            # Last epoch
            mlflow.log_metric(f"train_loss_epoch_{epochs}", history["train_loss"][-1])
            
            # Epoch with minimum loss
            min_loss_idx = history["train_loss"].index(min(history["train_loss"]))
            mlflow.log_metric(f"train_loss_epoch_{min_loss_idx + 1}", 
                            history["train_loss"][min_loss_idx])
            
            # Sample points throughout training
            sample_points = [int(epochs * 0.25), int(epochs * 0.5), int(epochs * 0.75)]
            for point in sample_points:
                if 1 < point < epochs:
                    mlflow.log_metric(f"train_loss_epoch_{point}", 
                                    history["train_loss"][point - 1])
    
    # Log validation loss metrics
    if history["val_loss"] and len(history["val_loss"]) > 0:
        mlflow.log_metric("final_val_loss", history["val_loss"][-1])
        mlflow.log_metric("min_val_loss", min(history["val_loss"]))
        
        # Log strategic epochs for validation loss
        epochs = len(history["val_loss"])
        if epochs <= 10:
            for i, loss in enumerate(history["val_loss"]):
                mlflow.log_metric(f"val_loss_epoch_{i + 1}", loss)
        else:
            # First epoch
            mlflow.log_metric("val_loss_epoch_1", history["val_loss"][0])
            # Last epoch
            mlflow.log_metric(f"val_loss_epoch_{epochs}", history["val_loss"][-1])
            
            # Epoch with minimum loss
            min_loss_idx = history["val_loss"].index(min(history["val_loss"]))
            mlflow.log_metric(f"val_loss_epoch_{min_loss_idx + 1}", 
                            history["val_loss"][min_loss_idx])
            
            # Sample points 
            sample_points = [int(epochs * 0.25), int(epochs * 0.5), int(epochs * 0.75)]
            for point in sample_points:
                if 1 < point < epochs:
                    mlflow.log_metric(f"val_loss_epoch_{point}", 
                                    history["val_loss"][point - 1])
    
    # Log epoch-specific metrics
    if "metrics" in history and history["metrics"] and len(history["metrics"]) > 0:
        for metric_name in history["metrics"][0].keys():
            # Extract values across epochs
            metric_values = [m.get(metric_name, None) for m in history["metrics"]]
            metric_values = [v for v in metric_values if v is not None]
            
            if metric_values:
                # Log final value
                mlflow.log_metric(f"final_{metric_name}", metric_values[-1])
                
                # For R², higher is better
                if metric_name == "r2":
                    best_value = max(metric_values)
                    best_epoch = metric_values.index(best_value) + 1
                    mlflow.log_metric(f"best_{metric_name}", best_value)
                    mlflow.log_metric(f"best_{metric_name}_epoch", best_epoch)
                # For error metrics, lower is better
                else:
                    best_value = min(metric_values)
                    best_epoch = metric_values.index(best_value) + 1
                    mlflow.log_metric(f"best_{metric_name}", best_value)
                    mlflow.log_metric(f"best_{metric_name}_epoch", best_epoch)


def initialize_train_evaluate_model(model_name, is_time_series, t_train, data_train, 
                               t_test, data_test, data_config, config, device, validation_split):
    """
    Initialize, train and evaluate a model.
    
    Args:
        model_name: Name of the model to initialize
        is_time_series: Whether this is a time series forecasting task
        t_train: Training time points
        data_train: Training data
        t_test: Test time points
        data_test: Test data
        data_config: Configuration of the dataset
        config: Main configuration dictionary
        device: Device to run the model on
        validation_split: Fraction of training data to use for validation
        
    Returns:
        tuple: (model, history, test_metrics, predictions, data_test, training_time)
    """
    # Check if this is a time series model
    is_ts_model = ("Forecaster" in model_name)
    
    # Initialize the model
    logger.info(f"Initializing model: {model_name}")
    if is_time_series and is_ts_model:
        # For time series models, pass extra parameters
        seq_len = config["hyperparameters"].get("seq_len", 96)
        pred_len = config["hyperparameters"].get("pred_len", 24)
        # Use per-timestep feature count for forecaster models
        model_kwargs = {
            "input_dim": data_config["input_dim"] if "input_dim" in data_config else 7,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "hidden_dim": config["hyperparameters"].get("hidden_dim", 64),
        }
        logger.info(f"Creating time series model with input_dim={model_kwargs['input_dim']}, "
                    f"seq_len={seq_len}, pred_len={pred_len}")
        model = get_model_by_name(model_name, **model_kwargs)
    else:
        # For non-time series models, use standard parameters but with correct dimensions from the data
        # This ensures input dimensions match exactly what we're passing to the model
        # TS data in flattened form already has shape [batch_size, features]
        
        # Inspect data shape to get actual dimensions
        if hasattr(data_train, 'shape') and len(data_train.shape) > 1:
            actual_input_dim = data_train.shape[1] - config["hyperparameters"].get("pred_len", 24)
            logger.info(f"Detected input dimension from data: {actual_input_dim}")
            input_dim = actual_input_dim
        elif "flattened_input_dim" in data_config:
            # Use flattened dimension from data config if available (for time series data)
            input_dim = data_config["flattened_input_dim"]
            logger.info(f"Using flattened input dimension from data_config: {input_dim}")
        else:
            # Use a reasonable input dimension as fallback
            input_dim = 1
            logger.info(f"Using input_dim=1 for synthetic dataset (was previously set to {config['hyperparameters'].get('input_dim', 96)} in config)")
            
        output_dim = config["hyperparameters"].get("output_dim", 24)
        hidden_dim = config["hyperparameters"].get("hidden_dim", 2048)
        
        model_kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
        }
        
        logger.info(f"Creating model with input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}")
        model = get_model_by_name(model_name, **model_kwargs)
        
        # We no longer need this monkey patching since we've fixed the forward methods in the models directly
        # This code is now redundant but keeping a comment for history
        # The models now handle dimension compatibility internally
    
    model.to(device)
    
    # Train the model
    logger.info(f"Training model with validation_split={validation_split}")
    train_start_time = time.time()
    
    try:
        # Enable offset tracking based on config and model type
        plot_offset_loss = (
            config.get("visualization", {}).get("offset_loss", False) and 
            "PhaseOffset" in model_name
        )
        if plot_offset_loss:
            logger.info(f"Enabling phase offset tracking for {model_name}")
        
        # Debug data shapes explicitly before training
        logger.info(f"Data shapes before training - t_train: {t_train.shape}, data_train: {data_train.shape}")
        
        # Convert 1D array to 2D column vector if needed
        if len(t_train.shape) == 1:
            logger.info("Reshaping 1D time array to 2D column vector")
            t_train = t_train.reshape(-1, 1)
            
        if len(data_train.shape) == 1:
            logger.info("Reshaping 1D data array to 2D column vector")
            data_train = data_train.reshape(-1, 1)
            
        logger.info(f"Data shapes after reshaping - t_train: {t_train.shape}, data_train: {data_train.shape}")
            
        history = train_model(
            model,
            t_train,
            data_train,
            config,
            device,
            validation_split=validation_split,
            plot_offset_loss=plot_offset_loss,
        )
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        # Create a minimal history to avoid downstream errors
        history = {
            "epochs": [1], 
            "train_loss": [float('inf')], 
            "val_loss": [float('inf')], 
            "metrics": [{"mse": float('inf')}],
            "epoch_times": [0.0]
        }
        
    # Calculate training time
    training_time = time.time() - train_start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model on test set
    logger.info("Evaluating model on test data")
    try:
        # Ensure test data has proper shape (convert 1D arrays to 2D column vectors if needed)
        if len(t_test.shape) == 1:
            logger.info("Reshaping 1D test time array to 2D column vector")
            t_test = t_test.reshape(-1, 1)
            
        if len(data_test.shape) == 1:
            logger.info("Reshaping 1D test data array to 2D column vector")
            data_test = data_test.reshape(-1, 1)
            
        logger.info(f"Test data shapes - t_test: {t_test.shape}, data_test: {data_test.shape}")
        
        if is_time_series:
            # For time series data, we need special handling
            test_metrics, predictions = evaluate_model(model, t_test, data_test, device, config)
            
            # Extract target values for visualization
            pred_len = config["hyperparameters"]["pred_len"]
            
            # Get only the target values for visualization
            ts_data_test = data_test.copy()
            data_test_targets = ts_data_test[:, -pred_len:]
            
            # Log shapes for debugging
            logger.info(f"Visualization shapes - Predictions: {predictions.shape}, Targets: {data_test_targets.shape}")
            
            # Update data_test to match predictions for visualization
            data_test = data_test_targets
        else:
            # Regular evaluation for non-time series data
            test_metrics, predictions = evaluate_model(model, t_test, data_test, device, None)
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        import traceback
        logger.error(f"Evaluation stack trace: {traceback.format_exc()}")
        # Create minimal metrics and predictions to avoid downstream errors
        test_metrics = {"mse": float('inf'), "rmse": float('inf'), "mae": float('inf'), 
                       "r2": -1.0, "mape": float('inf')}
        if is_time_series:
            pred_len = config["hyperparameters"]["pred_len"]
            # Create dummy predictions matching the expected shape
            predictions = np.zeros((len(data_test), pred_len))
        else:
            predictions = np.zeros_like(data_test)
    
    # Return model, history, metrics, predictions, processed test data, and training time
    return model, history, test_metrics, predictions, data_test, training_time


def load_and_transform_data(dataset_type, data_version, config):
    """
    Load and transform data based on dataset type and version.
    
    Args:
        dataset_type: Type of dataset to use (e.g., 'etth1', 'sin', 'mod')
        data_version: Version of data to use ('original', 'noisy', 'sparse')
        config: Configuration dictionary with hyperparameters
        
    Returns:
        tuple: (t_train, data_train, t_test, data_test, data_config, true_func)
    """
    is_time_series = dataset_type == "etth1"
    
    # Load base dataset
    logger.info(f"Loading {dataset_type} dataset...")
    if is_time_series:
        # For ETTh1 time series dataset
        seq_len = config["hyperparameters"].get("seq_len", 96)
        pred_len = config["hyperparameters"].get("pred_len", 24)
        target_col = config["hyperparameters"].get("target_col", "OT")

        # Get time series data
        t_train, data_train, t_test, data_test, data_config, true_func = get_etth1_data(
            seq_len=seq_len, pred_len=pred_len, target_col=target_col
        )
    else:
        # For synthetic data
        t_train, data_train, t_test, data_test, data_config, true_func = get_periodic_data(
            periodic_type=dataset_type,
            num_train_samples=config["hyperparameters"]["num_samples"],
            num_test_samples=config["hyperparameters"]["test_samples"],
        )
    
    # Apply transformations based on data_version
    if data_version == "original":
        logger.info("Using original data (no transformation).")
    elif data_version == "noisy":
        noise_level = config["hyperparameters"]["noise_level"]
        logger.info(f"Applying noise (level={noise_level}) to data.")

        if is_time_series:
            # For time series, only add noise to input features, not targets
            seq_len = config["hyperparameters"].get("seq_len", 96)
            pred_len = config["hyperparameters"].get("pred_len", 24)
            seq_features_len = seq_len * data_config["input_dim"] if "input_dim" in data_config else data_train.shape[1] - pred_len
            
            # Split inputs and targets
            data_train_input = data_train[:, :seq_features_len]
            data_train_target = data_train[:, seq_features_len:]
            data_test_input = data_test[:, :seq_features_len]
            data_test_target = data_test[:, seq_features_len:]

            # Add noise to inputs only
            data_train_input = add_noise(data_train_input, noise_level=noise_level)
            data_test_input = add_noise(data_test_input, noise_level=noise_level)

            # Recombine
            data_train = np.hstack((data_train_input, data_train_target))
            data_test = np.hstack((data_test_input, data_test_target))
        else:
            # For synthetic data, add noise to everything
            data_train = add_noise(data_train, noise_level=noise_level)
            data_test = add_noise(data_test, noise_level=noise_level)

    elif data_version == "sparse":
        sparsity_factor = config["hyperparameters"]["sparsity_factor"]
        logger.info(f"Making data sparse (factor={sparsity_factor}).")
        
        if is_time_series:
            # For time series, sparsify features but keep target values
            seq_len = config["hyperparameters"].get("seq_len", 96)
            pred_len = config["hyperparameters"].get("pred_len", 24)
            seq_features_len = seq_len * data_config["input_dim"] if "input_dim" in data_config else data_train.shape[1] - pred_len
            
            # Split inputs and targets
            data_train_input = data_train[:, :seq_features_len]
            data_train_target = data_train[:, seq_features_len:]
            data_test_input = data_test[:, :seq_features_len]
            data_test_target = data_test[:, seq_features_len:]
            
            # Apply sparsity only to input features (zero out values)
            mask_size = data_train_input.shape
            sparse_mask = np.random.rand(*mask_size) > (1 - 1/sparsity_factor)
            data_train_input_sparse = data_train_input.copy()
            data_train_input_sparse[sparse_mask] = 0
            
            # Apply same approach to test data
            mask_size_test = data_test_input.shape
            sparse_mask_test = np.random.rand(*mask_size_test) > (1 - 1/sparsity_factor)
            data_test_input_sparse = data_test_input.copy()
            data_test_input_sparse[sparse_mask_test] = 0
            
            # Recombine with targets
            data_train = np.hstack((data_train_input_sparse, data_train_target))
            data_test = np.hstack((data_test_input_sparse, data_test_target))
            
            logger.info(f"Created sparse time series data with {np.mean(sparse_mask)*100:.1f}% of input features zeroed")
        else:
            # For synthetic data, reduce the number of samples
            data_train, idx_train = make_sparse(data_train, sparsity_factor=sparsity_factor)
            data_test, idx_test = make_sparse(data_test, sparsity_factor=sparsity_factor)
            
            # Also shrink time arrays
            t_train = t_train[idx_train]
            t_test = t_test[idx_test]
    else:
        logger.error(f"Unknown data version: {data_version}")
        raise ValueError(f"Invalid data version: {data_version}")
    
    logger.info(f"Data loaded and transformed: {len(t_train)} train samples, {len(t_test)} test samples")
    return t_train, data_train, t_test, data_test, data_config, true_func


def run_experiment(model_name, dataset_type, data_version, config, experiment_id):
    """
    Runs a single experiment with the given configuration.

    Args:
        model_name: Name of the model to use
        dataset_type: Type of dataset to use
        data_version: Version of data to use (original, noisy, sparse)
        config: Configuration dictionary
        experiment_id: The MLflow experiment ID to log runs to

    Returns:
        str: The MLflow run ID of the experiment
    """
    run_id = None
    try:
        # Create descriptive run name
        run_name = f"{model_name}_{dataset_type}_{data_version}"

        # Start MLflow run
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

                # Determine which device to use
                device = select_device(config)
                mlflow.log_param("device", device.type)

                # Add the experiment batch name as a tag for grouping
                experiment_batch = config.get("experiment_name", "FAN_Model_Benchmark")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_id = f"{experiment_batch}_{timestamp}"
                mlflow.set_tag("batch_id", batch_id)
                mlflow.set_tag("experiment_batch", experiment_batch)
                
                # Load and transform data
                is_time_series = dataset_type == "etth1"
                t_train, data_train, t_test, data_test, data_config, true_func = load_and_transform_data(
                    dataset_type, data_version, config
                )
                
                # Log dataset sizes
                mlflow.log_metric("train_size", len(t_train))
                mlflow.log_metric("test_size", len(t_test))
                
                # Log time series specific parameters if applicable
                if is_time_series:
                    seq_len = config["hyperparameters"].get("seq_len", 96)
                    pred_len = config["hyperparameters"].get("pred_len", 24)
                    target_col = config["hyperparameters"].get("target_col", "OT")
                    mlflow.log_param("seq_len", seq_len)
                    mlflow.log_param("pred_len", pred_len)
                    mlflow.log_param("target_col", target_col)
                    mlflow.set_tag("task_type", "time_series_forecasting")
                
                # Log data transformation parameters if applicable
                if data_version == "noisy":
                    mlflow.log_param("noise_level", config["hyperparameters"]["noise_level"])
                elif data_version == "sparse":
                    mlflow.log_param("sparsity_factor", config["hyperparameters"]["sparsity_factor"])
                    mlflow.log_metric("train_size_after_sparse", len(t_train))
                    mlflow.log_metric("test_size_after_sparse", len(t_test))

                # Train and evaluate model
                validation_split = config["hyperparameters"].get("validation_split", 0.1)
                mlflow.log_param("validation_split", validation_split)
                
                # Initialize, train and evaluate the model
                model, history, test_metrics, predictions, data_test, training_time = initialize_train_evaluate_model(
                    model_name=model_name,
                    is_time_series=is_time_series,
                    t_train=t_train,
                    data_train=data_train,
                    t_test=t_test,
                    data_test=data_test,
                    data_config=data_config,
                    config=config,
                    device=device,
                    validation_split=validation_split
                )
                
                # Log training time
                mlflow.log_metric("training_time_seconds", training_time)
                mlflow.log_metric("last_epoch", history["epochs"][-1])
                
                # Log metrics from model evaluation
                log_metrics_and_history(test_metrics, history, training_time)

                # Generate visualization plots
                plots = generate_visualization_plots(
                    history=history,
                    model_name=model_name,
                    dataset_type=dataset_type,
                    data_version=data_version,
                    t_train=t_train, 
                    data_train=data_train,
                    t_test=t_test,
                    data_test=data_test,
                    predictions=predictions,
                    true_func=true_func,
                    config=config
                )

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


def generate_experiment_summary(run_ids, experiment_name, experiment_id):
    """
    Generate summary tables and visualizations for experiment runs.
    
    This function creates:
    1. A model summary table comparing performance across models
    2. Training loss comparison plots
    3. Efficiency visualizations for training speed
    4. Data type comparison tables
    5. Organized plot directories by dataset
    
    Args:
        run_ids: List of MLflow run IDs to include in the summary
        experiment_name: Name of the experiment
        experiment_id: MLflow experiment ID
    """
    # Generate model summary table
    logger.info("Generating model summary table...")
    summary_df = generate_model_summary_table(run_ids, experiment_name)
    if summary_df is not None:
        logger.info(f"Summary table generated with {len(summary_df)} runs")
    else:
        logger.warning("Failed to generate summary table")
    
    # Generate training loss comparison plots
    logger.info("Generating loss comparison plots...")
    train_loss_plot = plot_losses_by_epoch_comparison(
        run_ids=run_ids, 
        metric_name="train_loss", 
        include_validation=False
    )
    
    # Generate training efficiency visualizations
    logger.info("Generating training efficiency visualizations...")
    efficiency_plots = create_all_efficiency_visualizations(
        run_ids, experiment_name
    )
    
    # Generate concise visualizations - make sure to filter by current experiment ID
    logger.info("Generating enhanced visualizations with efficiency plots...")
    all_plots = create_enhanced_visualizations(
        run_ids, 
        experiment_id, 
        experiment_name,
        with_efficiency_plots=efficiency_plots,
        max_epochs=None,  # Show all epochs
        current_experiment_only=True  # Only use runs from the current experiment
    )
    
    # Log visualization summary
    if all_plots:
        logger.info(f"Generated {len(all_plots)} visualization plots total")
        if efficiency_plots:
            logger.info(f"Includes {len(efficiency_plots)} efficiency plots")
    
    # Generate data type comparison table
    logger.info("Generating data type comparison table...")
    dt_comparison_df, dt_comparison_path = generate_data_type_comparison_table(run_ids, experiment_name)
    if dt_comparison_df is not None:
        logger.info(f"Data type comparison table generated with {len(dt_comparison_df)} model/dataset combinations")
    else:
        logger.warning("Failed to generate data type comparison table")
    
    # Organize plots by dataset
    logger.info("Organizing plots by dataset...")
    all_generated_plots = []
    
    # Collect plots from all plot directories
    for path in glob.glob("plots/*.png"):
        all_generated_plots.append(path)
    for path in glob.glob("plots/*/*.png"):
        all_generated_plots.append(path)
    
    # Organize into subdirectories by dataset/type
    if all_generated_plots:
        organized_paths = organize_plots_by_dataset_and_type(all_generated_plots, "plots/by_dataset")
        logger.info(f"Organized {len(organized_paths)} plots by dataset")
        
        # Log the organized plots to MLflow
        if experiment_id:
            with mlflow.start_run(
                run_name=f"PlotsByDataset-{time.strftime('%Y%m%d-%H%M%S')}", 
                experiment_id=experiment_id
            ):
                # Log each plot individually
                plot_count = 0
                for root, dirs, files in os.walk("plots/by_dataset"):
                    for file in files:
                        if file.endswith('.png'):
                            file_path = os.path.join(root, file)
                            # Create simplified artifact path
                            artifact_path = os.path.relpath(os.path.dirname(file_path), "plots/by_dataset")
                            if artifact_path == '.':
                                artifact_path = None
                            try:
                                mlflow.log_artifact(file_path, artifact_path)
                                plot_count += 1
                            except Exception as e:
                                logger.error(f"Failed to log {file_path}: {e}")
                
                mlflow.set_tag("artifact_type", "organized_plots")
                mlflow.set_tag("is_summary", "true")
                mlflow.set_tag("plot_count", str(plot_count))
                logger.info(f"Logged {plot_count} organized plots to MLflow")


def save_run_ids(run_ids, experiment_name):
    """
    Save run IDs to a JSON file for later analysis.

    Args:
        run_ids: List of run IDs
        experiment_name: Name of the experiment
    """
    if not run_ids:
        logger.warning("No run IDs to save")
        return

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Generate filename with timestamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = f"results/run_ids_{experiment_name.replace(' ', '_')}_{timestr}.json"

    # Save to file
    with open(filename, "w") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "run_ids": run_ids,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": os.environ.get("USER", "keirparker"),
                "count": len(run_ids),
            },
            f,
            indent=2,
        )

    logger.info(f"Saved {len(run_ids)} run IDs to {filename}")

def setup_mlflow_experiment(experiment_name):
    """
    Set up an MLflow experiment, creating it if it doesn't exist.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        str: The experiment ID
    """
    # Get or create the experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    # Create if doesn't exist
    if experiment is None:
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            # If we can't create it, use the default experiment
            experiment_id = "0"  # Default experiment ID in MLflow
            logger.warning("Using default experiment (ID: 0)")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
    return experiment_id


def main(config_path='configs/ts_config.yml', verbose=False):
    """
    Main entry point. Loads configuration and runs all specified experiments.
    
    Args:
        config_path: Path to the configuration YAML file
        verbose: Whether to enable verbose logging
    """
    # Setup logging with appropriate level
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(log_level)

    # Log script start
    logger.info(f"ML experiment framework started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Running as user: {os.environ.get('USER', 'keirparker')}")
    logger.info(f"Using configuration from: {config_path}")

    try:
        # Load configuration from specified path
        config = load_config(config_path)
        setup_environment(config)

        # Get experiment parameters
        models_to_run = config["models_to_run"]
        dataset_types = config["datasets_to_run"]
        data_versions = config["data_versions"]

        # Get experiment name and set up MLflow
        experiment_name = config.get("experiment_name", "FAN_Model_Benchmark")
        experiment_id = setup_mlflow_experiment(experiment_name)
        
        # Calculate number of experiments
        total_experiments = len(models_to_run) * len(dataset_types) * len(data_versions)
        logger.info(f"Starting {total_experiments} experiments")

        # Store all run IDs for comparison
        all_run_ids = []

        # Run all combinations
        for model_name in models_to_run:
            for dataset_type in dataset_types:
                for data_version in data_versions:
                    logger.info(f"Running experiment: model={model_name}, dataset={dataset_type}, version={data_version}")
                    try:
                        run_id = run_experiment(
                            model_name,
                            dataset_type,
                            data_version,
                            config,
                            experiment_id,
                        )
                        all_run_ids.append(run_id)
                    except Exception as e:
                        logger.error(f"Error in experiment: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

        logger.info("All experiments completed")

        # Save run IDs for later analysis
        save_run_ids(all_run_ids, experiment_name)

        # Generate summary and visualizations
        if all_run_ids:
            generate_experiment_summary(all_run_ids, experiment_name, experiment_id)



    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Clean up any lingering MLflow runs
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            logger.warning(f"Found active run at end of execution (ID: {run_id}). Closing it.")
            mlflow.end_run()

        # Log execution completion
        logger.info(f"ML experiment framework finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



def parse_args():
    """
    Parse command line arguments for the runner script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fourier Analysis Networks (FANs) Experimentation Framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        default='configs/config.yml',
        help='Path to the configuration YAML file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config, verbose=args.verbose)
