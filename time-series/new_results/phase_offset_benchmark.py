#!/usr/bin/env python
"""
PhaseOffset Model Benchmark Script

This script specifically benchmarks the PhaseOffsetTransformerForecaster model against others
and provides detailed metrics.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from loguru import logger
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from time_series.models import get_model_by_name
from time_series.data import create_time_series_loaders, TimeSeriesDataset
from time_series.data.ts_dataset import load_multivariate_dataset

# Define datasets and focus models
DATASETS = ["electricity", "traffic", "solar-energy"]
MODELS = [
    "TransformerForecaster",  # Baseline
    "PhaseOffsetTransformerForecaster",  # Focus model
]

def setup_logger():
    """Configure the logger."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    logger.add(
        os.path.join(log_dir, f"po_benchmark_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="10 MB"
    )
    logger.info("Logger configured successfully")

def examine_checkpoint(model_name, dataset_name):
    """
    Examines the content of a checkpoint file and prints detailed information.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
    """
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "checkpoints",
        f"{model_name}_{dataset_name}",
        "model.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        logger.info(f"Examining checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Print basic info
        logger.info(f"Checkpoint file size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Inspect model state
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            logger.info(f"Model state contains {len(model_state)} layers/parameters")
            logger.info(f"Sample keys: {list(model_state.keys())[:5]}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
            logger.info(f"Total parameters: {total_params:,}")
        
        # Check config
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            config = checkpoint["config"]
            logger.info(f"Configuration keys: {list(config.keys())}")
            
            if "hyperparameters" in config:
                hp = config["hyperparameters"]
                logger.info("Hyperparameters:")
                for key, value in hp.items():
                    logger.info(f"  {key}: {value}")
        
        # Check metrics
        if "val_loss" in checkpoint:
            logger.info(f"Validation loss: {checkpoint['val_loss']}")
        
        if "history" in checkpoint and isinstance(checkpoint["history"], dict):
            history = checkpoint["history"]
            if "val_loss" in history and len(history["val_loss"]) > 0:
                min_val_loss = min(history["val_loss"])
                min_epoch = history["val_loss"].index(min_val_loss) + 1
                logger.info(f"Best validation loss: {min_val_loss} at epoch {min_epoch}")
                
                # Plot if there are multiple epochs
                if len(history["val_loss"]) > 1:
                    logger.info(f"Training history contains {len(history['val_loss'])} epochs")
    
    except Exception as e:
        logger.error(f"Error examining checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())

def load_model_with_checkpoint(model_name, dataset_name, dataset_config, dataset_shape):
    """
    Loads a model with its checkpoint for detailed inspection.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        dataset_config: Configuration for the dataset
        dataset_shape: Shape of the dataset
        
    Returns:
        model: Loaded model
    """
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "checkpoints",
        f"{model_name}_{dataset_name}",
        "model.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint for model inspection: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Create model with appropriate dimensions
        input_dim = dataset_shape[1] if len(dataset_shape) > 1 else 1
        output_dim = input_dim
        horizon = dataset_config.get("horizon", 24)
        
        # Create model parameters
        model_params = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "horizon": horizon,
            "dropout": dataset_config.get("dropout", 0.1),
        }
        
        # Add model-specific parameters
        if "PhaseOffsetTransformerForecaster" in model_name:
            model_params.update({
                "d_model": dataset_config.get("d_model", 64),
                "fan_dim": dataset_config.get("hidden_dim", 64),
                "nhead": dataset_config.get("n_heads", 4),
                "num_layers": dataset_config.get("n_layers", 2),
                "use_checkpointing": False,  # Disable for inspection
            })
        elif "TransformerForecaster" in model_name:
            model_params.update({
                "hidden_dim": dataset_config.get("d_model", 64),
                "nhead": dataset_config.get("n_heads", 4),
                "num_layers": dataset_config.get("n_layers", 2),
            })
        
        # Create model
        logger.info(f"Creating model '{model_name}' with params: {model_params}")
        model = get_model_by_name(model_name, **model_params)
        
        # Load state
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Model state loaded successfully")
        else:
            logger.warning("No model_state_dict found in checkpoint")
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict) and any(k.endswith(".weight") for k in checkpoint[key].keys()):
                    logger.info(f"Trying to load state from '{key}'")
                    try:
                        model.load_state_dict(checkpoint[key])
                        logger.info(f"Successfully loaded state from '{key}'")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load state from '{key}': {e}")
            
        model.eval()
        return model
    
    except Exception as e:
        logger.error(f"Error loading model with checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def evaluate_model_detailed(model, test_loader, device):
    """
    Evaluate model with detailed metrics and diagnostics.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: The device to run evaluation on
        
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: Model predictions
        targets: Ground truth targets
    """
    if model is None or test_loader is None:
        return None, None, None
    
    model.eval()
    criterion = torch.nn.MSELoss(reduction='none')
    mae_criterion = torch.nn.L1Loss(reduction='none')
    
    all_preds = []
    all_targets = []
    all_losses = []
    all_maes = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            try:
                # Run with extra debugging
                logger.debug(f"Input shape: {data.shape}, Target shape: {target.shape}")
                
                # Forward pass
                output = model(data)
                
                # Calculate losses
                loss = criterion(output, target)  # Per-element loss
                mae = mae_criterion(output, target)  # Per-element MAE
                
                # Store detailed results
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_losses.append(loss.cpu().numpy())
                all_maes.append(mae.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.error(f"Model: {type(model).__name__}, Input shape: {data.shape}, Target shape: {target.shape}")
    
    if not all_preds:
        logger.error("No predictions were generated")
        return None, None, None
    
    # Combine and reshape
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    losses = np.vstack(all_losses)
    maes = np.vstack(all_maes)
    
    # Overall metrics
    mse = np.mean(losses)
    rmse = np.sqrt(mse)
    mae = np.mean(maes)
    
    # Calculate per-feature metrics
    feature_mse = np.mean(losses, axis=0)
    feature_rmse = np.sqrt(feature_mse)
    feature_mae = np.mean(maes, axis=0)
    
    # Calculate horizon-based metrics (how error changes with forecast horizon)
    horizon_mse = np.mean(losses, axis=(0, 2))  # Average across batches and features
    horizon_rmse = np.sqrt(horizon_mse)
    horizon_mae = np.mean(maes, axis=(0, 2))
    
    logger.info(f"Overall metrics - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "feature_mse": feature_mse,
        "feature_rmse": feature_rmse,
        "feature_mae": feature_mae,
        "horizon_mse": horizon_mse,
        "horizon_rmse": horizon_rmse,
        "horizon_mae": horizon_mae
    }, predictions, targets

def visualize_predictions(predictions, targets, model_name, dataset_name, output_dir):
    """
    Create visualizations of model predictions vs actual values.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        model_name: Name of the model
        dataset_name: Name of the dataset
        output_dir: Output directory for visualizations
        
    Returns:
        plot_path: Path to the generated plot
    """
    if predictions is None or targets is None:
        return None
    
    try:
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Determine features to visualize (sample a few)
        num_features = predictions.shape[2]
        num_samples = min(5, predictions.shape[0])
        sample_features = [0]  # Always include first feature
        
        # Add more feature indices if available
        if num_features > 1:
            sample_features.extend([num_features // 2, num_features - 1])  # Middle and last
        
        # Get number of time steps in prediction
        horizon = predictions.shape[1]
        
        # Create a grid of subplots - one row per feature, one column per sample
        fig, axes = plt.subplots(len(sample_features), num_samples, 
                                figsize=(15, 4 * len(sample_features)), 
                                squeeze=False)
        
        # Plot each sample and feature
        for i, feature_idx in enumerate(sample_features):
            for j in range(num_samples):
                ax = axes[i, j]
                
                # Plot actual values
                ax.plot(range(horizon), targets[j, :, feature_idx], 
                        'o-', label='Actual', color='blue', alpha=0.7)
                
                # Plot predictions
                ax.plot(range(horizon), predictions[j, :, feature_idx], 
                        'x--', label='Prediction', color='red', alpha=0.7)
                
                # Add labels
                if i == 0:
                    ax.set_title(f"Sample {j+1}")
                if j == 0:
                    ax.set_ylabel(f"Feature {feature_idx}")
                if i == len(sample_features) - 1:
                    ax.set_xlabel("Time Step")
                
                # Add legend to first plot only
                if i == 0 and j == 0:
                    ax.legend()
        
        # Add overall title
        plt.suptitle(f"{model_name} Predictions vs Actual on {dataset_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save plot
        plot_path = os.path.join(plots_dir, f"predictions_{model_name}_{dataset_name}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved prediction visualization to {plot_path}")
        return plot_path
    
    except Exception as e:
        logger.error(f"Error generating prediction visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def visualize_forecast_horizon_errors(metrics, model_names, dataset_name, output_dir):
    """
    Visualize how forecast errors change with horizon length.
    
    Args:
        metrics: Dictionary of model metrics
        model_names: Names of models to compare
        dataset_name: Name of the dataset
        output_dir: Output directory for visualizations
        
    Returns:
        plot_path: Path to the generated plot
    """
    try:
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot horizon RMSE for each model
        for model_name in model_names:
            if model_name in metrics and 'horizon_rmse' in metrics[model_name]:
                horizon_rmse = metrics[model_name]['horizon_rmse']
                plt.plot(range(1, len(horizon_rmse) + 1), horizon_rmse, 
                        'o-', label=f"{model_name}", linewidth=2, markersize=4)
        
        # Add labels and title
        plt.xlabel("Forecast Horizon (Time Steps)", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.title(f"Forecast Error by Horizon - {dataset_name}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(plots_dir, f"horizon_errors_{dataset_name}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved horizon error visualization to {plot_path}")
        return plot_path
    
    except Exception as e:
        logger.error(f"Error generating horizon error visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_phase_offset_benchmark():
    """Run the PhaseOffset benchmark."""
    setup_logger()
    logger.info("Starting Phase Offset model benchmark")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if hasattr(torch, 'mps') and torch.backends.mps.is_available() else 
                             "cpu")
        logger.info(f"Using device: {device}")
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            dataset_configs = config.get("dataset_config", {})
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            dataset_configs = {}
        
        # Results container for all models and datasets
        all_results = {}
        all_metrics = {}
        
        # First, just examine checkpoints for troubleshooting
        logger.info("=== CHECKPOINT EXAMINATION ===")
        for model_name in MODELS:
            for dataset_name in DATASETS:
                examine_checkpoint(model_name, dataset_name)
        
        # Now run the benchmark
        logger.info("=== RUNNING BENCHMARK ===")
        for dataset_name in DATASETS:
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Load dataset
            try:
                dataset_config = dataset_configs.get(dataset_name, {})
                normalize = True
                norm_method = dataset_config.get("norm_method", "standard")
                if norm_method.lower() == "none":
                    normalize = False
                
                memory_efficient = dataset_config.get("memory_efficient", True)
                
                logger.info(f"Loading dataset {dataset_name} with normalization={normalize}")
                data = load_multivariate_dataset(
                    dataset_name, 
                    normalize=normalize,
                    memory_efficient=memory_efficient
                )
                
                if data is None:
                    logger.error(f"Failed to load dataset: {dataset_name}")
                    continue
                    
                logger.info(f"Loaded {dataset_name} with shape {data.shape}")
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_name}: {e}")
                continue
            
            # Get dataset parameters
            lookback = dataset_config.get("lookback", 96)
            horizon = dataset_config.get("horizon", 24)
            batch_size = dataset_config.get("batch_size", 32)
            
            # Create data loaders
            try:
                test_size = 0.2  # Last 20% for testing
                val_size = 0.1   # 10% for validation
                
                # Create data loaders
                _, val_loader, test_loader = create_time_series_loaders(
                    data,
                    lookback=lookback,
                    horizon=horizon,
                    stride=1,
                    batch_size=batch_size,
                    test_size=test_size,
                    val_size=val_size,
                    num_workers=4,
                    pin_memory=True,
                    shuffle=False  # No need to shuffle for evaluation
                )
                
                logger.info(f"Created data loaders for {dataset_name} with lookback={lookback}, horizon={horizon}")
            except Exception as e:
                logger.error(f"Error creating data loaders for {dataset_name}: {e}")
                continue
            
            # Dictionary to store results for this dataset
            dataset_results = {}
            dataset_metrics = {}
            
            # Evaluate each model
            for model_name in MODELS:
                logger.info(f"Evaluating {model_name} on {dataset_name}")
                
                try:
                    # Load model with checkpoint
                    model = load_model_with_checkpoint(
                        model_name, dataset_name, dataset_config, data.shape
                    )
                    
                    if model is None:
                        logger.warning(f"Skipping {model_name} on {dataset_name}: failed to load model")
                        continue
                    
                    # Move model to device
                    model = model.to(device)
                    
                    # Detailed evaluation
                    metrics, predictions, targets = evaluate_model_detailed(
                        model, test_loader, device
                    )
                    
                    if metrics is None:
                        logger.warning(f"Skipping {model_name} on {dataset_name}: evaluation failed")
                        continue
                    
                    # Store results
                    dataset_results[model_name] = {
                        "mse": metrics["mse"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"]
                    }
                    
                    # Store detailed metrics
                    dataset_metrics[model_name] = metrics
                    
                    # Visualize predictions
                    visualize_predictions(
                        predictions, targets, model_name, dataset_name, output_dir
                    )
                    
                    logger.info(f"Evaluated {model_name} on {dataset_name}: " +
                               f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Visualize horizon errors for this dataset
            if len(dataset_metrics) > 1:  # Only if we have multiple models
                visualize_forecast_horizon_errors(
                    dataset_metrics, MODELS, dataset_name, output_dir
                )
            
            # Store dataset results
            all_results[dataset_name] = dataset_results
            all_metrics[dataset_name] = dataset_metrics
        
        # Generate summary
        generate_summary_report(all_results, all_metrics, output_dir)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in benchmark: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

def generate_summary_report(results, metrics, output_dir):
    """
    Generate a summary report of the benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        metrics: Dictionary of detailed metrics
        output_dir: Output directory
    """
    if not results:
        logger.error("No results to generate summary")
        return
    
    try:
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Define model display names
        model_display_names = {
            "TransformerForecaster": "Transformer",
            "PhaseOffsetTransformerForecaster": "PhaseOffsetTransformer",
        }
        
        # Dataset display names
        dataset_display_names = {
            "electricity": "Electricity",
            "traffic": "Traffic",
            "solar-energy": "Solar"
        }
        
        # Create HTML report
        html_path = os.path.join(output_dir, f"phase_offset_report_{timestamp}.html")
        
        with open(html_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>PhaseOffsetTransformer Benchmark Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .section {{ margin-top: 30px; margin-bottom: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .highlight {{ background-color: #d5f5e3; }}
        .better {{ color: green; font-weight: bold; }}
        .worse {{ color: red; }}
        .image-container {{ max-width: 900px; margin: 20px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .footer {{ margin-top: 40px; font-size: 0.8em; color: #7f8c8d; text-align: right; }}
    </style>
</head>
<body>
    <h1>PhaseOffsetTransformer Benchmark Report</h1>
    <p>Benchmark generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <p>This report compares the PhaseOffsetTransformer model with the baseline Transformer model.</p>
        
        <table>
            <tr>
                <th>Dataset</th>
                <th>Model</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>Comparison to Baseline</th>
            </tr>
""")
            
            # Add rows for each dataset and model
            for dataset_name in sorted(results.keys()):
                dataset_results = results[dataset_name]
                display_dataset = dataset_display_names.get(dataset_name, dataset_name.capitalize())
                
                # Find baseline metrics
                baseline_rmse = None
                baseline_mae = None
                if "TransformerForecaster" in dataset_results:
                    baseline_rmse = dataset_results["TransformerForecaster"]["rmse"]
                    baseline_mae = dataset_results["TransformerForecaster"]["mae"]
                
                for model_name in MODELS:
                    if model_name not in dataset_results:
                        continue
                        
                    display_model = model_display_names.get(model_name, model_name)
                    model_results = dataset_results[model_name]
                    
                    # Calculate improvement
                    comparison_text = "-"
                    if model_name != "TransformerForecaster" and baseline_rmse is not None:
                        rmse_change = (model_results["rmse"] - baseline_rmse) / baseline_rmse * 100
                        
                        if rmse_change < 0:
                            comparison_text = f'<span class="better">{rmse_change:.1f}% better</span>'
                        else:
                            comparison_text = f'<span class="worse">{rmse_change:.1f}% worse</span>'
                    
                    # First row for dataset gets rowspan
                    if model_name == MODELS[0]:
                        f.write(f"""
            <tr>
                <td rowspan="{len([m for m in MODELS if m in dataset_results])}">{display_dataset}</td>
                <td>{display_model}</td>
                <td>{model_results["rmse"]:.4f}</td>
                <td>{model_results["mae"]:.4f}</td>
                <td>{comparison_text}</td>
            </tr>
""")
                    else:
                        f.write(f"""
            <tr>
                <td>{display_model}</td>
                <td>{model_results["rmse"]:.4f}</td>
                <td>{model_results["mae"]:.4f}</td>
                <td>{comparison_text}</td>
            </tr>
""")
            
            # Add detailed findings
            f.write(f"""
        </table>
    </div>
    
    <div class="section">
        <h2>Detailed Findings</h2>
        
        <h3>Performance by Forecast Horizon</h3>
        <p>The following plots show how forecast error changes with the forecast horizon:</p>
        
        <div class="image-container">
            <img src="plots/horizon_errors_electricity_{timestamp}.png" alt="Horizon Errors - Electricity">
        </div>
        
        <div class="image-container">
            <img src="plots/horizon_errors_traffic_{timestamp}.png" alt="Horizon Errors - Traffic">
        </div>
        
        <div class="image-container">
            <img src="plots/horizon_errors_solar-energy_{timestamp}.png" alt="Horizon Errors - Solar">
        </div>
        
        <h3>Sample Predictions</h3>
        <p>The following visualizations compare predictions with actual values:</p>
""")
            
            # Add prediction visualizations
            for model_name in MODELS:
                display_model = model_display_names.get(model_name, model_name)
                
                f.write(f"""
        <h4>{display_model}</h4>
""")
                
                for dataset_name in DATASETS:
                    display_dataset = dataset_display_names.get(dataset_name, dataset_name.capitalize())
                    
                    f.write(f"""
        <div class="image-container">
            <img src="plots/predictions_{model_name}_{dataset_name}_{timestamp}.png" alt="Predictions - {display_model} on {display_dataset}">
        </div>
""")
            
            # Add conclusion
            f.write("""
    </div>
    
    <div class="section">
        <h2>Conclusion</h2>
        <p>Based on the benchmark results, we observe that:</p>
        <ul>
""")
            
            # Add dataset-specific conclusions
            for dataset_name in sorted(results.keys()):
                dataset_results = results[dataset_name]
                display_dataset = dataset_display_names.get(dataset_name, dataset_name.capitalize())
                
                if "TransformerForecaster" in dataset_results and "PhaseOffsetTransformerForecaster" in dataset_results:
                    baseline_rmse = dataset_results["TransformerForecaster"]["rmse"]
                    po_rmse = dataset_results["PhaseOffsetTransformerForecaster"]["rmse"]
                    
                    if po_rmse < baseline_rmse:
                        improvement = (baseline_rmse - po_rmse) / baseline_rmse * 100
                        f.write(f"""
            <li>For the <strong>{display_dataset}</strong> dataset, PhaseOffsetTransformer performs <span class="better">{improvement:.1f}% better</span> than the baseline Transformer.</li>
""")
                    else:
                        decline = (po_rmse - baseline_rmse) / baseline_rmse * 100
                        f.write(f"""
            <li>For the <strong>{display_dataset}</strong> dataset, PhaseOffsetTransformer performs <span class="worse">{decline:.1f}% worse</span> than the baseline Transformer.</li>
""")
            
            # Add general conclusion about PhaseOffsetTransformer
            overall_better = 0
            overall_datasets = 0
            
            for dataset_name in results:
                dataset_results = results[dataset_name]
                if "TransformerForecaster" in dataset_results and "PhaseOffsetTransformerForecaster" in dataset_results:
                    overall_datasets += 1
                    if dataset_results["PhaseOffsetTransformerForecaster"]["rmse"] < dataset_results["TransformerForecaster"]["rmse"]:
                        overall_better += 1
            
            if overall_datasets > 0:
                if overall_better > overall_datasets / 2:
                    f.write(f"""
            <li>Overall, PhaseOffsetTransformer outperforms the baseline Transformer on {overall_better} out of {overall_datasets} datasets.</li>
""")
                elif overall_better == overall_datasets / 2:
                    f.write(f"""
            <li>Overall, PhaseOffsetTransformer performs comparably to the baseline Transformer, winning on {overall_better} out of {overall_datasets} datasets.</li>
""")
                else:
                    f.write(f"""
            <li>Overall, the baseline Transformer outperforms PhaseOffsetTransformer on {overall_datasets - overall_better} out of {overall_datasets} datasets.</li>
""")
            
            # Close HTML
            f.write("""
        </ul>
    </div>
    
    <div class="footer">
        <p>Generated with phase_offset_benchmark.py | Analysis of Phase Offset vs Standard Transformer Models</p>
    </div>
</body>
</html>
""")
        
        logger.info(f"Summary report saved to {html_path}")
        
        # Also save a CSV with results
        csv_path = os.path.join(output_dir, f"phase_offset_results_{timestamp}.csv")
        
        with open(csv_path, "w") as f:
            f.write("Dataset,Model,MSE,RMSE,MAE\n")
            
            for dataset_name in sorted(results.keys()):
                dataset_results = results[dataset_name]
                
                for model_name in sorted(dataset_results.keys()):
                    metrics = dataset_results[model_name]
                    f.write(f"{dataset_name},{model_name},{metrics['mse']:.6f},{metrics['rmse']:.6f},{metrics['mae']:.6f}\n")
        
        logger.info(f"Results CSV saved to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    sys.exit(run_phase_offset_benchmark())