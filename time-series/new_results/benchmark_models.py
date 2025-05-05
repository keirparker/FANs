#!/usr/bin/env python
"""
Model Benchmark Script

This script loads pre-trained models from checkpoints and evaluates them on test data,
ensuring proper benchmarking without data leakage.
"""

import os
import sys
import json
import time
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

# Define datasets and models
DATASETS = ["electricity", "traffic", "solar-energy"]
# Focus especially on PhaseOffsetTransformerForecaster
MODELS = [
    "TransformerForecaster",
    "PhaseOffsetTransformerForecaster",  # Focus model 1
    "FANTransformerForecaster"
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
        os.path.join(log_dir, f"benchmark_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="10 MB"
    )
    logger.info("Logger configured successfully")

def load_checkpoint(model_name, dataset_name):
    """
    Load a model checkpoint.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        
    Returns:
        model: The loaded model
        checkpoint: The checkpoint data
    """
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "checkpoints",
        f"{model_name}_{dataset_name}",
        "model.pt"
    )
    
    # Add model verification
    logger.info(f"Verifying model {model_name} on dataset {dataset_name}")
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None, None
    
    try:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        # Try different device mappings if one fails
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        except Exception as e1:
            logger.warning(f"Error loading with CPU, trying 'mps': {e1}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location="mps")
            except Exception as e2:
                logger.warning(f"Error loading with 'mps', trying 'cuda': {e2}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cuda")
                except Exception as e3:
                    logger.warning(f"Error loading with 'cuda', trying map_location='cpu': {e3}")
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Print more detailed information about the checkpoint
        logger.info(f"Checkpoint loaded for {model_name} on {dataset_name}")
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Inspect checkpoint content in more detail to verify it's correct
        if "val_loss" in checkpoint:
            logger.info(f"Validation loss from checkpoint: {checkpoint['val_loss']}")
        
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            if "hyperparameters" in checkpoint["config"]:
                hp = checkpoint["config"]["hyperparameters"]
                logger.info(f"Model hyperparameters: dropout={hp.get('dropout')}, " +
                           f"hidden_dim={hp.get('hidden_dim')}, lr={hp.get('lr')}")
        
        # Extract configuration
        config = checkpoint.get("config", {})
        
        return None, checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def create_model(model_name, dataset, dataset_config):
    """
    Create a model instance with proper configuration.
    
    Args:
        model_name: Name of the model
        dataset: The dataset
        dataset_config: Configuration for the dataset
        
    Returns:
        model: The created model
    """
    try:
        # Determine dimensions from dataset
        input_dim = dataset.shape[1] if len(dataset.shape) > 1 else 1
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
        if "FANForecaster" in model_name or "FANGatedForecaster" in model_name:
            model_params.update({
                "hidden_dim": dataset_config.get("hidden_dim", 64),
                "n_fan_layers": dataset_config.get("n_layers", 2),
            })
        elif "FANTransformerForecaster" in model_name or "PhaseOffsetTransformerForecaster" in model_name:
            model_params.update({
                "d_model": dataset_config.get("d_model", 64),
                "fan_dim": dataset_config.get("hidden_dim", 64),
                "nhead": dataset_config.get("n_heads", 4),
                "num_layers": dataset_config.get("n_layers", 2),
                "use_checkpointing": dataset_config.get("use_checkpoint", True),
            })
        elif "TransformerForecaster" in model_name:
            model_params.update({
                "hidden_dim": dataset_config.get("d_model", 64),
                "nhead": dataset_config.get("n_heads", 4),
                "num_layers": dataset_config.get("n_layers", 2),
            })
        
        logger.info(f"Creating model '{model_name}' with params: {model_params}")
        model = get_model_by_name(model_name, **model_params)
        return model
    except Exception as e:
        logger.error(f"Error creating model {model_name}: {e}")
        return None

def load_model_state(model, checkpoint):
    """
    Load model state from checkpoint.
    
    Args:
        model: The model to load state into
        checkpoint: The checkpoint data
        
    Returns:
        model: The model with loaded state
    """
    if model is None or checkpoint is None:
        return None
    
    try:
        # Check different possible key names for model state
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            model_state = checkpoint["state_dict"]
        else:
            # Try to find any dict that might contain the model state
            for key, value in checkpoint.items():
                if isinstance(value, dict) and any(k.endswith("weight") for k in value.keys()):
                    model_state = value
                    logger.info(f"Found model state in key: {key}")
                    break
            else:
                logger.warning("No model state found in checkpoint")
                logger.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
                return None
        
        model.load_state_dict(model_state)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model state: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test data.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: The device to run evaluation on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if model is None or test_loader is None:
        return None
    
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
    }

def benchmark_models():
    """
    Benchmark all models on all datasets.
    
    Returns:
        results: Dictionary of benchmark results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch, 'mps') and torch.backends.mps.is_available() else 
                         "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset-specific configurations
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_configs = config.get("dataset_config", {})
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        dataset_configs = {}
    
    # Results container
    all_results = {}
    
    # Benchmark each model on each dataset
    for dataset_name in DATASETS:
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Load dataset
        try:
            # Use dataset-specific configuration if available
            dataset_config = dataset_configs.get(dataset_name, {})
            
            # Default to standard normalization
            normalize = True
            norm_method = dataset_config.get("norm_method", "standard")
            if norm_method.lower() == "none":
                normalize = False
            
            # Use memory-efficient loading
            memory_efficient = dataset_config.get("memory_efficient", True)
            
            # Load dataset
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
        
        # Create data loaders with fixed train/val/test split to ensure fair comparison
        try:
            test_size = 0.2  # Last 20% for testing
            val_size = 0.1   # 10% for validation
            
            # Use only validation and test data
            # Skip creating train_loader as we're only evaluating pre-trained models
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
        
        # Benchmark each model
        dataset_results = {}
        
        for model_name in MODELS:
            logger.info(f"Benchmarking {model_name} on {dataset_name}")
            
            try:
                # Load checkpoint
                _, checkpoint = load_checkpoint(model_name, dataset_name)
                
                if checkpoint is None:
                    logger.warning(f"Skipping {model_name} on {dataset_name}: checkpoint not found")
                    continue
                
                # Create model
                model = create_model(model_name, data, dataset_config)
                
                if model is None:
                    logger.warning(f"Skipping {model_name} on {dataset_name}: failed to create model")
                    continue
                
                # Load model state
                model = load_model_state(model, checkpoint)
                
                if model is None:
                    logger.warning(f"Skipping {model_name} on {dataset_name}: failed to load model state")
                    continue
                
                # Move model to device
                model = model.to(device)
                
                # Evaluate model
                metrics = evaluate_model(model, test_loader, device)
                
                if metrics is None:
                    logger.warning(f"Skipping {model_name} on {dataset_name}: evaluation failed")
                    continue
                
                dataset_results[model_name] = metrics
                logger.info(f"Benchmarked {model_name} on {dataset_name}: {metrics}")
            except Exception as e:
                logger.error(f"Error benchmarking {model_name} on {dataset_name}: {e}")
                continue
        
        all_results[dataset_name] = dataset_results
    
    return all_results

def generate_performance_table(results):
    """
    Generate performance comparison tables.
    
    Args:
        results: Dictionary of benchmark results
        
    Returns:
        outputs: Dictionary of output file paths
    """
    if not results:
        logger.error("No results to generate tables from")
        return {}
    
    outputs = {}
    
    # Create output directory
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Define model display names (shorter, more academic naming)
    model_display_names = {
        "TransformerForecaster": "Transformer",
        "FANTransformerForecaster": "FAN-T",
        "PhaseOffsetTransformerForecaster": "PO-T",
        "FANForecaster": "FAN",
        "FANGatedForecaster": "FAN-G",
        "FANGatedTransformerForecaster": "FAN-GT"
    }
    
    # Dataset display names (capitalized)
    dataset_display_names = {
        "electricity": "Electricity",
        "traffic": "Traffic",
        "solar-energy": "Solar"
    }
    
    # Convert results to DataFrame format for easier processing
    rows = []
    
    for dataset_name, dataset_results in results.items():
        for model_name, metrics in dataset_results.items():
            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"]
            }
            rows.append(row)
    
    if not rows:
        logger.error("No valid rows to generate tables from")
        return {}
    
    df = pd.DataFrame(rows)
    
    # Save raw results to CSV
    csv_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    outputs["csv"] = csv_path
    logger.info(f"Saved raw results to {csv_path}")
    
    # Calculate best model for each dataset and metric
    best_models = {}
    for dataset in df["Dataset"].unique():
        dataset_df = df[df["Dataset"] == dataset]
        
        best_models[dataset] = {
            "MSE": dataset_df.loc[dataset_df["MSE"].idxmin()]["Model"] if not dataset_df.empty else None,
            "RMSE": dataset_df.loc[dataset_df["RMSE"].idxmin()]["Model"] if not dataset_df.empty else None,
            "MAE": dataset_df.loc[dataset_df["MAE"].idxmin()]["Model"] if not dataset_df.empty else None
        }
    
    # Generate markdown table
    markdown = "# Model Benchmark Results\n\n"
    markdown += f"Benchmark ran on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Create main table
    markdown += "## Performance by Model and Dataset\n\n"
    markdown += "| Model | "
    
    # Add dataset columns for each metric
    for dataset in sorted(results.keys()):
        display_name = dataset_display_names.get(dataset, dataset.capitalize())
        markdown += f"{display_name} (RMSE) | {display_name} (MAE) | "
    
    # Add average columns
    markdown += "Avg RMSE | Avg MAE |\n"
    
    # Add separator
    markdown += "|-------|"
    for _ in range(len(results)):
        markdown += "-------|-------|"
    markdown += "-------|-------|\n"
    
    # Calculate model averages
    model_avgs = {}
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        model_avgs[model] = {
            "RMSE": model_df["RMSE"].mean(),
            "MAE": model_df["MAE"].mean()
        }
    
    # Find overall best model
    best_avg_rmse_model = min(model_avgs.items(), key=lambda x: x[1]["RMSE"])[0] if model_avgs else None
    best_avg_mae_model = min(model_avgs.items(), key=lambda x: x[1]["MAE"])[0] if model_avgs else None
    
    # Add rows for each model
    # Sort models by average RMSE (best first)
    sorted_models = sorted(model_avgs.keys(), key=lambda x: model_avgs[x]["RMSE"]) if model_avgs else df["Model"].unique()
    
    for model in sorted_models:
        # Get display name
        display_name = model_display_names.get(model, model)
        markdown += f"| {display_name} | "
        
        # Add metrics for each dataset
        for dataset in sorted(results.keys()):
            dataset_results = results.get(dataset, {})
            model_results = dataset_results.get(model, {})
            
            if model_results:
                rmse = model_results.get("rmse", None)
                mae = model_results.get("mae", None)
                
                # Bold if best for this dataset
                if best_models.get(dataset, {}).get("RMSE") == model:
                    rmse_str = f"**{rmse:.4f}**" if rmse is not None else "-"
                else:
                    rmse_str = f"{rmse:.4f}" if rmse is not None else "-"
                
                if best_models.get(dataset, {}).get("MAE") == model:
                    mae_str = f"**{mae:.4f}**" if mae is not None else "-"
                else:
                    mae_str = f"{mae:.4f}" if mae is not None else "-"
                
                markdown += f"{rmse_str} | {mae_str} | "
            else:
                markdown += "- | - | "
        
        # Add average metrics
        avg_rmse = model_avgs.get(model, {}).get("RMSE", None)
        avg_mae = model_avgs.get(model, {}).get("MAE", None)
        
        # Bold if best average
        if model == best_avg_rmse_model:
            avg_rmse_str = f"**{avg_rmse:.4f}**" if avg_rmse is not None else "-"
        else:
            avg_rmse_str = f"{avg_rmse:.4f}" if avg_rmse is not None else "-"
        
        if model == best_avg_mae_model:
            avg_mae_str = f"**{avg_mae:.4f}**" if avg_mae is not None else "-"
        else:
            avg_mae_str = f"{avg_mae:.4f}" if avg_mae is not None else "-"
        
        markdown += f"{avg_rmse_str} | {avg_mae_str} |\n"
    
    # Add notes
    markdown += "\n**Note:** Best values are in **bold**.\n"
    
    # Save markdown table
    md_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.md")
    with open(md_path, "w") as f:
        f.write(markdown)
    outputs["markdown"] = md_path
    logger.info(f"Saved markdown results to {md_path}")
    
    # Generate HTML table with added visualizations
    generate_html_report(results, model_avgs, best_models, output_dir, timestamp, outputs)
    
    return outputs

def generate_html_report(results, model_avgs, best_models, output_dir, timestamp, outputs):
    """
    Generate HTML report with visualizations.
    
    Args:
        results: Dictionary of benchmark results
        model_avgs: Dictionary of model averages
        best_models: Dictionary of best models for each dataset
        output_dir: Output directory
        timestamp: Timestamp string
        outputs: Dictionary of output file paths to update
    """
    try:
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Model display names
        model_display_names = {
            "TransformerForecaster": "Transformer",
            "FANTransformerForecaster": "FAN-T",
            "PhaseOffsetTransformerForecaster": "PO-T",
            "FANForecaster": "FAN",
            "FANGatedForecaster": "FAN-G",
            "FANGatedTransformerForecaster": "FAN-GT"
        }
        
        # Dataset display names
        dataset_display_names = {
            "electricity": "Electricity",
            "traffic": "Traffic",
            "solar-energy": "Solar"
        }
        
        # Generate RMSE comparison plot
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plot
        plot_data = []
        for dataset, dataset_results in results.items():
            for model, metrics in dataset_results.items():
                plot_data.append({
                    "Dataset": dataset_display_names.get(dataset, dataset.capitalize()),
                    "Model": model_display_names.get(model, model),
                    "RMSE": metrics["rmse"]
                })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Create grouped bar chart
            ax = sns.barplot(x="Dataset", y="RMSE", hue="Model", data=plot_df)
            
            # Enhance appearance
            plt.title("RMSE by Model and Dataset", fontsize=16)
            plt.xlabel("Dataset", fontsize=14)
            plt.ylabel("RMSE (lower is better)", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(title="Model", title_fontsize=12, fontsize=10)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            
            # Save plot
            rmse_plot_path = os.path.join(plots_dir, f"rmse_comparison_{timestamp}.png")
            plt.savefig(rmse_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            outputs["rmse_plot"] = rmse_plot_path
            logger.info(f"Saved RMSE comparison plot to {rmse_plot_path}")
        
        # Generate model ranking heatmap
        plt.figure(figsize=(10, 6))
        
        # Prepare ranking data
        ranking_data = {}
        
        for dataset in results:
            ranking_data[dataset] = {}
            
            # Get RMSE values for all models on this dataset
            rmse_values = [(model, metrics["rmse"]) for model, metrics in results[dataset].items()]
            
            # Sort by RMSE (lower is better)
            rmse_values.sort(key=lambda x: x[1])
            
            # Assign ranks
            for rank, (model, _) in enumerate(rmse_values, 1):
                display_name = model_display_names.get(model, model)
                ranking_data[dataset][display_name] = rank
        
        if ranking_data:
            # Convert to DataFrame
            rank_df = pd.DataFrame(ranking_data).T
            
            # Create heatmap
            ax = sns.heatmap(rank_df, annot=True, cmap="YlGnBu_r", fmt="d", cbar_kws={"label": "Rank (lower is better)"})
            
            # Enhance appearance
            plt.title("Model Rankings by Dataset (RMSE)", fontsize=16)
            plt.xlabel("Model", fontsize=14)
            plt.ylabel("Dataset", fontsize=14)
            plt.tight_layout()
            
            # Save plot
            rank_plot_path = os.path.join(plots_dir, f"ranking_heatmap_{timestamp}.png")
            plt.savefig(rank_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            outputs["rank_plot"] = rank_plot_path
            logger.info(f"Saved ranking heatmap to {rank_plot_path}")
        
        # Generate overall performance plot
        plt.figure(figsize=(10, 6))
        
        # Prepare average performance data
        if model_avgs:
            models = []
            rmse_avgs = []
            mae_avgs = []
            
            for model, avgs in model_avgs.items():
                models.append(model_display_names.get(model, model))
                rmse_avgs.append(avgs["RMSE"])
                mae_avgs.append(avgs["MAE"])
            
            # Sort by RMSE
            sort_idx = np.argsort(rmse_avgs)
            models = [models[i] for i in sort_idx]
            rmse_avgs = [rmse_avgs[i] for i in sort_idx]
            mae_avgs = [mae_avgs[i] for i in sort_idx]
            
            # Create grouped bar chart
            x = np.arange(len(models))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, rmse_avgs, width, label="RMSE")
            ax.bar(x + width/2, mae_avgs, width, label="MAE")
            
            # Enhance appearance
            ax.set_title("Average Model Performance", fontsize=16)
            ax.set_xlabel("Model", fontsize=14)
            ax.set_ylabel("Error (lower is better)", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            
            # Save plot
            avg_plot_path = os.path.join(plots_dir, f"average_performance_{timestamp}.png")
            plt.savefig(avg_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            outputs["avg_plot"] = avg_plot_path
            logger.info(f"Saved average performance plot to {avg_plot_path}")
        
        # Generate HTML report
        html_path = os.path.join(output_dir, f"benchmark_report_{timestamp}.html")
        
        # Determine best models by average metrics
        best_avg_rmse_model = None
        best_avg_mae_model = None
        
        if model_avgs:
            best_avg_rmse = float('inf')
            best_avg_mae = float('inf')
            
            for model, metrics in model_avgs.items():
                if metrics["RMSE"] < best_avg_rmse:
                    best_avg_rmse = metrics["RMSE"]
                    best_avg_rmse_model = model
                if metrics["MAE"] < best_avg_mae:
                    best_avg_mae = metrics["MAE"]
                    best_avg_mae_model = model
        
        with open(html_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Benchmark Report</title>
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
        .image-container {{ max-width: 900px; margin: 20px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .footer {{ margin-top: 40px; font-size: 0.8em; color: #7f8c8d; text-align: right; }}
    </style>
</head>
<body>
    <h1>Time Series Model Benchmark Report</h1>
    <p>Benchmark generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <p>This report compares the performance of different time series forecasting models on multiple datasets.</p>
        
        <h3>Model Descriptions</h3>
        <ul>
            <li><strong>Transformer</strong>: Standard Transformer model for time series forecasting</li>
            <li><strong>FAN-T</strong>: Transformer with Fourier Amplitude Neural (FAN) layers</li>
            <li><strong>PO-T</strong>: Transformer with Phase Offset mechanism</li>
            <li><strong>FAN</strong>: Pure Fourier Amplitude Neural model</li>
            <li><strong>FAN-G</strong>: Gated Fourier Amplitude Neural model</li>
            <li><strong>FAN-GT</strong>: Gated Transformer with FAN layers</li>
        </ul>
        
        <h3>Datasets</h3>
        <ul>
            <li><strong>Electricity</strong>: Electricity consumption data (321 features)</li>
            <li><strong>Traffic</strong>: Traffic occupancy rates data (862 features)</li>
            <li><strong>Solar</strong>: Solar power production data (137 features)</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Performance Comparison</h2>
        
        <div class="image-container">
            <h3>RMSE by Model and Dataset</h3>
            <img src="plots/rmse_comparison_{timestamp}.png" alt="RMSE Comparison">
        </div>
        
        <div class="image-container">
            <h3>Model Rankings (RMSE)</h3>
            <img src="plots/ranking_heatmap_{timestamp}.png" alt="Model Rankings">
            <p><em>Note: Lower rank (1) indicates better performance.</em></p>
        </div>
        
        <div class="image-container">
            <h3>Average Model Performance</h3>
            <img src="plots/average_performance_{timestamp}.png" alt="Average Performance">
        </div>
    </div>
    
    <div class="section">
        <h2>Detailed Results</h2>
        
        <table>
            <tr>
                <th>Model</th>
""")
            
            # Add dataset headers
            for dataset in sorted(results.keys()):
                display_name = dataset_display_names.get(dataset, dataset.capitalize())
                f.write(f"                <th>{display_name} (RMSE)</th>\n")
                f.write(f"                <th>{display_name} (MAE)</th>\n")
            
            f.write(f"                <th>Avg RMSE</th>\n")
            f.write(f"                <th>Avg MAE</th>\n")
            f.write(f"            </tr>\n")
            
            # Add rows for each model
            sorted_models = sorted(model_avgs.keys(), key=lambda x: model_avgs[x]["RMSE"]) if model_avgs else []
            
            for model in sorted_models:
                display_name = model_display_names.get(model, model)
                f.write(f"            <tr>\n")
                f.write(f"                <td>{display_name}</td>\n")
                
                # Add metrics for each dataset
                for dataset in sorted(results.keys()):
                    dataset_results = results.get(dataset, {})
                    model_results = dataset_results.get(model, {})
                    
                    if model_results:
                        rmse = model_results.get("rmse")
                        mae = model_results.get("mae")
                        
                        # Highlight if best
                        is_best_rmse = best_models.get(dataset, {}).get("RMSE") == model
                        is_best_mae = best_models.get(dataset, {}).get("MAE") == model
                        
                        rmse_class = ' class="highlight"' if is_best_rmse else ''
                        mae_class = ' class="highlight"' if is_best_mae else ''
                        
                        f.write(f"                <td{rmse_class}>{rmse:.4f}</td>\n")
                        f.write(f"                <td{mae_class}>{mae:.4f}</td>\n")
                    else:
                        f.write(f"                <td>-</td>\n")
                        f.write(f"                <td>-</td>\n")
                
                # Add average metrics
                avg_rmse = model_avgs.get(model, {}).get("RMSE")
                avg_mae = model_avgs.get(model, {}).get("MAE")
                
                # Highlight if best average
                is_best_avg_rmse = model == best_avg_rmse_model
                is_best_avg_mae = model == best_avg_mae_model
                
                avg_rmse_class = ' class="highlight"' if is_best_avg_rmse else ''
                avg_mae_class = ' class="highlight"' if is_best_avg_mae else ''
                
                f.write(f"                <td{avg_rmse_class}>{avg_rmse:.4f}</td>\n")
                f.write(f"                <td{avg_mae_class}>{avg_mae:.4f}</td>\n")
                f.write(f"            </tr>\n")
            
            f.write(f"""        </table>
        <p><em>Note: Highlighted cells indicate best performance for each dataset and metric.</em></p>
    </div>
    
    <div class="section">
        <h2>Conclusion</h2>
        <p>Based on the benchmark results, we can observe that:</p>
        <ul>
""")
            
            # Add conclusions based on results
            if best_avg_rmse_model:
                best_name = model_display_names.get(best_avg_rmse_model, best_avg_rmse_model)
                f.write(f"            <li><strong>{best_name}</strong> has the best overall performance in terms of RMSE.</li>\n")
            
            if best_avg_mae_model and best_avg_mae_model != best_avg_rmse_model:
                best_mae_name = model_display_names.get(best_avg_mae_model, best_avg_mae_model)
                f.write(f"            <li><strong>{best_mae_name}</strong> has the best overall performance in terms of MAE.</li>\n")
            
            # Add observations about model types
            fan_models = [m for m in sorted_models if "FAN" in m]
            regular_models = [m for m in sorted_models if "FAN" not in m]
            
            if fan_models and regular_models and model_avgs:
                fan_avg_rmse = np.mean([model_avgs[m]["RMSE"] for m in fan_models])
                regular_avg_rmse = np.mean([model_avgs[m]["RMSE"] for m in regular_models])
                
                if fan_avg_rmse < regular_avg_rmse:
                    f.write(f"            <li>FAN-based models generally outperform regular transformer models.</li>\n")
                else:
                    f.write(f"            <li>Regular transformer models generally perform better than FAN-based models.</li>\n")
            
            # Add dataset-specific observations
            for dataset in results:
                if dataset in best_models and best_models[dataset].get("RMSE"):
                    best_model = best_models[dataset]["RMSE"]
                    best_name = model_display_names.get(best_model, best_model)
                    display_dataset = dataset_display_names.get(dataset, dataset.capitalize())
                    f.write(f"            <li>For the <strong>{display_dataset}</strong> dataset, <strong>{best_name}</strong> achieves the best performance.</li>\n")
            
            f.write(f"""        </ul>
    </div>
    
    <div class="footer">
        <p>Generated with benchmark_models.py | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
""")
        
        outputs["html"] = html_path
        logger.info(f"Saved HTML report to {html_path}")
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main function."""
    setup_logger()
    logger.info("Starting model benchmark")
    
    try:
        # Run benchmarks
        results = benchmark_models()
        
        if not results:
            logger.error("Benchmark failed to produce results. Exiting.")
            return 1
        
        # Generate tables and visualizations
        outputs = generate_performance_table(results)
        
        if not outputs:
            logger.error("Failed to generate performance tables. Exiting.")
            return 1
        
        logger.info("Benchmark completed successfully!")
        logger.info(f"Output files: {outputs}")
        
        # Print path to HTML report
        if "html" in outputs:
            logger.info(f"HTML report: {outputs['html']}")
            print(f"\nHTML report generated: {outputs['html']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in benchmark: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())