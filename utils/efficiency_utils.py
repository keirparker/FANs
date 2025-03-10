import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
import mlflow
from datetime import datetime
from typing import List, Dict, Any, Optional


def track_epoch_times(history):
    """
    Adds epoch timestamps to history dictionary.
    
    Args:
        history: Training history dictionary
        
    Returns:
        Updated history dictionary with epoch_times
    """
    if 'epoch_times' not in history:
        # Initialize epoch timing tracking
        history['epoch_times'] = []
        history['epoch_start_time'] = time.time()
    
    # Record time for this epoch
    current_time = time.time()
    epoch_time = current_time - history.get('epoch_start_time', current_time)
    history['epoch_times'].append(epoch_time)
    
    # Reset start time for next epoch
    history['epoch_start_time'] = current_time
    
    return history


def create_efficiency_comparison(run_ids, experiment_name, experiment_dir=None, timestamp=None):
    """
    Create visualizations to compare model training efficiency.
    
    Args:
        run_ids: List of MLflow run IDs
        experiment_name: Name of the experiment for titling
        experiment_dir: Directory to save plots in (optional)
        timestamp: Timestamp for consistent file naming (optional)
    
    Returns:
        List of paths to generated plots
    """
    client = mlflow.tracking.MlflowClient()
    plots = []
    
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    # Create experiment directory if not provided
    if experiment_dir is None:
        experiment_dir = f"plots/efficiency_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger.info(f"Created efficiency visualization directory: {experiment_dir}")
    
    # Create dataframe for comparison
    df_rows = []
    
    for run_id in run_ids:
        if not run_id:
            continue
            
        try:
            run = client.get_run(run_id)
            run_data = run.data
            
            # Extract key metrics and parameters
            model_name = run_data.params.get("model", "Unknown")
            dataset = run_data.params.get("dataset_type", "Unknown")
            data_version = run_data.params.get("data_version", "Unknown")
            
            # Training metrics
            training_time = run_data.metrics.get("training_time_seconds", np.nan)
            final_train_loss = run_data.metrics.get("final_train_loss", np.nan)
            final_val_loss = run_data.metrics.get("final_val_loss", np.nan)
            
            # Test metrics
            test_mse = run_data.metrics.get("test_mse", np.nan)
            test_rmse = run_data.metrics.get("test_rmse", np.nan)
            test_mae = run_data.metrics.get("test_mae", np.nan)
            test_r2 = run_data.metrics.get("test_r2", np.nan)
            
            # Number of parameters and epochs
            num_epochs = float(run_data.params.get("epochs", np.nan))
            
            # Add to dataframe rows
            df_rows.append({
                "Model": model_name,
                "Dataset": dataset,
                "Version": data_version,
                "Training Time (s)": training_time,
                "Final Train Loss": final_train_loss,
                "Final Val Loss": final_val_loss,
                "Test MSE": test_mse,
                "Test RMSE": test_rmse,
                "Test MAE": test_mae,
                "Test R²": test_r2,
                "Epochs": num_epochs,
                "Training Speed (s/epoch)": training_time / num_epochs if not np.isnan(training_time) and not np.isnan(num_epochs) else np.nan,
                "Model ID": f"{model_name} ({dataset}, {data_version})"
            })
            
        except Exception as e:
            logger.error(f"Error extracting data for run {run_id}: {e}")
    
    if not df_rows:
        logger.warning("No data available to create efficiency comparison")
        return plots
    
    # Create DataFrame
    df = pd.DataFrame(df_rows)
    
    # 1. Plot training time comparison
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Sort for better visualization
    df_sorted = df.sort_values("Training Time (s)", ascending=False)
    
    # Use hue parameter to prevent warning and for better coloring
    ax = sns.barplot(x="Model ID", y="Training Time (s)", data=df_sorted, hue="Model", palette="viridis", legend=False)
    plt.title(f"Training Time Comparison - {experiment_name}", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Add values on top of bars
    for i, v in enumerate(df_sorted["Training Time (s)"]):
        if not np.isnan(v):
            ax.text(i, v + 1, f"{v:.1f}s", ha='center')
    
    # Save to experiment directory with consistent timestamp
    train_time_plot = f"{experiment_dir}/training_time_comparison_{timestamp}.png"
    plt.savefig(train_time_plot)
    plt.close()
    plots.append(train_time_plot)
    
    # 2. Plot training speed (seconds per epoch)
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values("Training Speed (s/epoch)")
    ax = sns.barplot(x="Model ID", y="Training Speed (s/epoch)", data=df_sorted, hue="Model ID", palette="rocket", legend=False)
    plt.title(f"Training Speed Comparison (Lower is Better) - {experiment_name}", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Add values on top of bars
    for i, v in enumerate(df_sorted["Training Speed (s/epoch)"]):
        if not np.isnan(v):
            ax.text(i, v + 0.1, f"{v:.1f}s", ha='center')
    
    train_speed_plot = f"{experiment_dir}/training_speed_comparison_{timestamp}.png"
    plt.savefig(train_speed_plot)
    plt.close()
    plots.append(train_speed_plot)
    
    # 3. Plot final metrics vs training time (bubble chart)
    plt.figure(figsize=(12, 8))
    
    # Filter out rows with missing data
    df_filtered = df.dropna(subset=["Training Time (s)", "Test RMSE", "Test R²"])
    
    if not df_filtered.empty:
        # Create scatter plot with size proportional to R²
        # Adjust R² values to always be positive for sizing (adding 1 to negative values)
        size_values = df_filtered["Test R²"].copy()
        size_values[size_values < 0] = 0  # Set negative values to 0
        size_values = (size_values + 1) * 200  # Scale for visibility

        # Create a dictionary to map models to colors for consistent coloring
        unique_models = df_filtered["Model"].unique()
        color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))
        model_to_color = {model: color_map[i] for i, model in enumerate(unique_models)}
        
        # Create scatter plot with points colored by model type
        scatter_points = []
        legend_labels = []
        
        for model_name in unique_models:
            model_data = df_filtered[df_filtered["Model"] == model_name]
            
            # Plot each model with its own color
            scatter = plt.scatter(
                model_data["Training Time (s)"], 
                model_data["Test RMSE"],
                s=size_values[df_filtered["Model"] == model_name], 
                color=model_to_color[model_name],
                alpha=0.7,
                edgecolors='k',
                label=model_name
            )
            scatter_points.append(scatter)
            legend_labels.append(model_name)
        
        plt.title(f"Model Efficiency Comparison - {experiment_name}", fontsize=16)
        plt.xlabel("Training Time (seconds)", fontsize=12)
        plt.ylabel("Test RMSE (lower is better)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add a proper legend with model names
        plt.legend(scatter_points, legend_labels, 
                  title="Model Types",
                  loc="upper right", 
                  fontsize=10)
        
        # Add R² legend explanation
        plt.figtext(0.15, 0.02, "Note: Bubble size represents model performance (R²)", 
                    fontsize=10, ha='left')
        
        efficiency_plot = f"{experiment_dir}/model_efficiency_comparison_{timestamp}.png"
        plt.savefig(efficiency_plot, bbox_inches='tight')
        plt.close()
        plots.append(efficiency_plot)
    
    # 4. Create a performance summary table
    plt.figure(figsize=(16, 6))
    plt.axis('off')
    
    # Create a summary table focusing on efficiency metrics
    summary_df = df[["Model", "Training Time (s)", "Training Speed (s/epoch)", 
                     "Test RMSE", "Test R²"]].copy()
    
    # Sort by performance
    summary_df = summary_df.sort_values("Test RMSE")
    
    # Format the numeric columns
    for col in summary_df.columns[1:]:
        if summary_df[col].dtype == float:
            summary_df[col] = summary_df[col].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
    
    # Create table
    table = plt.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(summary_df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    plt.title(f"Model Performance Summary - {experiment_name}", 
              fontsize=16, pad=20)
    
    summary_plot = f"{experiment_dir}/model_summary_table_{timestamp}.png"
    plt.savefig(summary_plot, bbox_inches='tight')
    plt.close()
    plots.append(summary_plot)
    
    # Individual MLflow logging is now handled by the parent function
    # to ensure all plots are logged together
    
    return plots


def plot_training_progress_comparison(run_ids, metric_name="train_loss", smoothing=0.2, experiment_dir=None, timestamp=None):
    """
    Create a plot comparing training progress across different models.
    
    Args:
        run_ids: List of MLflow run IDs
        metric_name: Name of metric to plot
        smoothing: Amount of smoothing to apply to the curves
        experiment_dir: Directory to save the plot in
        timestamp: Timestamp for consistent file naming
        
    Returns:
        Path to the generated plot
    """
    client = mlflow.tracking.MlflowClient()
    
    # Create dict to store metric data by model
    model_data = {}
    
    # Extract data from MLflow
    for run_id in run_ids:
        if not run_id:
            continue
            
        try:
            run = client.get_run(run_id)
            
            # Get model name and other metadata
            model_name = run.data.params.get("model", "Unknown")
            dataset = run.data.params.get("dataset_type", "Unknown")
            data_version = run.data.params.get("data_version", "Unknown")
            
            # Initialize model entry
            label = f"{model_name} ({dataset}, {data_version})"
            if label not in model_data:
                model_data[label] = {"epochs": [], "values": []}
            
            # Extract metric values by epoch
            for key, value in run.data.metrics.items():
                if key.startswith(f"{metric_name}_epoch_"):
                    try:
                        epoch = int(key.split("_")[-1])
                        model_data[label]["epochs"].append(epoch)
                        model_data[label]["values"].append(value)
                    except (ValueError, IndexError):
                        continue
                        
            # Sort by epoch
            if model_data[label]["epochs"]:
                sorted_data = sorted(zip(model_data[label]["epochs"], model_data[label]["values"]))
                model_data[label]["epochs"] = [item[0] for item in sorted_data]
                model_data[label]["values"] = [item[1] for item in sorted_data]
                
        except Exception as e:
            logger.error(f"Error processing run {run_id}: {e}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot data for each model
    legend_entries = []
    colors = plt.cm.tab10.colors
    
    for i, (label, data) in enumerate(model_data.items()):
        if not data["epochs"] or not data["values"]:
            continue
            
        # Apply smoothing if needed
        if smoothing > 0 and len(data["values"]) > 3:
            values = pd.Series(data["values"]).ewm(alpha=1-smoothing).mean().values
        else:
            values = data["values"]
            
        # Plot the line
        line = plt.plot(
            data["epochs"], 
            values, 
            '-o', 
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=5,
            alpha=0.8,
            label=label
        )[0]
        
        legend_entries.append(line)
    
    # Add title and labels
    metric_display = metric_name.replace("_", " ").title()
    plt.title(f"Training Progress Comparison - {metric_display}", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(metric_display, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Add legend
    if legend_entries:
        plt.legend(fontsize=10)
    
    # Use the same experiment directory as the main function for consistency
    if not os.path.exists(experiment_dir):
        experiment_dir = f"plots/efficiency_{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(experiment_dir, exist_ok=True)
    
    # Save the plot, using the same timestamp for consistent naming
    plot_path = f"{experiment_dir}/training_progress_{metric_name}_{timestamp}.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    # Individual MLflow logging is now handled by the parent function
    # to ensure all plots are logged together
    
    return plot_path


def create_accuracy_complexity_plot(run_ids, experiment_name, experiment_dir=None, timestamp=None):
    """
    Create a visualization comparing model accuracy vs. complexity (parameter count).
    
    Args:
        run_ids: List of MLflow run IDs
        experiment_name: Name of the experiment for titling
        experiment_dir: Directory to save the plot in
        timestamp: Timestamp for consistent file naming
        
    Returns:
        Path to the generated plot
    """
    client = mlflow.tracking.MlflowClient()
    
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    # Create experiment directory if not provided
    if experiment_dir is None:
        experiment_dir = f"plots/efficiency_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
    
    # Extract data for plotting
    model_data = []
    
    for run_id in run_ids:
        if not run_id:
            continue
            
        try:
            run = client.get_run(run_id)
            
            # Extract model info
            model_name = run.data.params.get("model", "Unknown")
            
            # Get parameter count (if recorded)
            param_count = run.data.metrics.get("param_count", None)
            if param_count is None:
                # Try alternative names
                param_count = run.data.metrics.get("parameter_count", None)
                if param_count is None:
                    param_count = run.data.metrics.get("model_parameters", None)
            
            # If no direct parameter count, try to calculate from model type
            if param_count is None:
                # Some approximate values based on model types - adjust as needed
                if "Transformer" in model_name:
                    hidden_dim = int(run.data.params.get("hidden_dim", "64"))
                    num_layers = int(run.data.params.get("num_layers", "2"))
                    param_count = hidden_dim * hidden_dim * num_layers * 4
                elif "LSTM" in model_name:
                    hidden_dim = int(run.data.params.get("hidden_dim", "64"))
                    param_count = hidden_dim * hidden_dim * 4
                else:
                    # Skip if we can't determine parameter count
                    continue
            
            # Get accuracy metrics
            test_rmse = run.data.metrics.get("test_rmse", None)
            test_mae = run.data.metrics.get("test_mae", None)
            test_r2 = run.data.metrics.get("test_r2", None)
            
            if test_rmse is not None and param_count is not None:
                model_data.append({
                    "Model": model_name,
                    "Parameters": param_count,
                    "RMSE": test_rmse,
                    "MAE": test_mae if test_mae is not None else np.nan,
                    "R²": test_r2 if test_r2 is not None else np.nan
                })
                
        except Exception as e:
            logger.error(f"Error processing run {run_id} for accuracy-complexity plot: {e}")
    
    if not model_data:
        logger.warning("No model data available for accuracy-complexity plot")
        return None
    
    # Create dataframe and plot
    df = pd.DataFrame(model_data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create unique colors for each model type
    unique_models = df["Model"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    model_colors = {model: colors[i] for i, model in enumerate(unique_models)}
    
    # Draw points with different colors per model type
    for model in unique_models:
        model_df = df[df["Model"] == model]
        plt.scatter(
            model_df["Parameters"], 
            model_df["RMSE"],
            s=100,
            label=model,
            color=model_colors[model],
            alpha=0.7,
            edgecolors='black'
        )
    
    # Add best fit line (optional)
    if len(df) > 2:
        try:
            # Simple log curve fit
            x = np.log10(df["Parameters"])
            y = df["RMSE"]
            
            # Filter out NaN values
            mask = ~np.isnan(x) & ~np.isnan(y)
            if sum(mask) > 2:
                x = x[mask]
                y = y[mask]
                
                # Fit a polynomial
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Generate points for the line
                x_line = np.linspace(min(x), max(x), 100)
                y_line = p(x_line)
                
                # Plot trend line
                plt.plot(10**x_line, y_line, 'r--', alpha=0.5, 
                        label=f"Trend line (slope: {z[0]:.3f})")
        except Exception as e:
            logger.error(f"Error fitting trend line: {e}")
    
    plt.xscale('log')  # Log scale for parameter count
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel("Model Complexity (number of parameters)", fontsize=12)
    plt.ylabel("Error (RMSE, lower is better)", fontsize=12)
    plt.title(f"Model Accuracy vs. Complexity - {experiment_name}", fontsize=16)
    plt.legend(title="Model Type")
    
    # Add annotations for key models
    # Choose best and worst models to annotate
    if len(df) > 2:
        best_model = df.loc[df["RMSE"].idxmin()]
        worst_model = df.loc[df["RMSE"].idxmax()]
        
        # Add annotations
        plt.annotate(
            f"{best_model['Model']}\nRMSE: {best_model['RMSE']:.4f}",
            (best_model["Parameters"], best_model["RMSE"]),
            xytext=(15, 0),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        )
        
        plt.annotate(
            f"{worst_model['Model']}\nRMSE: {worst_model['RMSE']:.4f}",
            (worst_model["Parameters"], worst_model["RMSE"]),
            xytext=(15, 0),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        )
    
    # Save the plot
    plot_path = f"{experiment_dir}/accuracy_vs_complexity_{timestamp}.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    return plot_path


def create_convergence_speed_plot(run_ids, experiment_name, experiment_dir=None, timestamp=None):
    """
    Create a visualization showing how quickly each model converges to a good solution.
    
    Args:
        run_ids: List of MLflow run IDs
        experiment_name: Name of the experiment for titling
        experiment_dir: Directory to save the plot in
        timestamp: Timestamp for consistent file naming
        
    Returns:
        Path to the generated plot
    """
    client = mlflow.tracking.MlflowClient()
    
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    # Create experiment directory if not provided
    if experiment_dir is None:
        experiment_dir = f"plots/efficiency_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
    
    # Dictionary to store convergence data by model
    convergence_data = {}
    
    # Target loss improvement threshold to consider "converged" (e.g., 95% of final improvement)
    convergence_threshold = 0.95
    
    for run_id in run_ids:
        if not run_id:
            continue
            
        try:
            run = client.get_run(run_id)
            model_name = run.data.params.get("model", "Unknown")
            
            # Get all the loss metrics by epoch
            loss_metrics = {}
            for key, value in run.data.metrics.items():
                if key.startswith("val_loss_epoch_"):
                    try:
                        epoch = int(key.split("_")[-1])
                        loss_metrics[epoch] = value
                    except (ValueError, IndexError):
                        continue
            
            # If we don't have enough loss metrics, skip this model
            if len(loss_metrics) < 5:  # Need at least a few epochs to analyze convergence
                continue
                
            # Sort by epoch
            epochs = sorted(loss_metrics.keys())
            losses = [loss_metrics[e] for e in epochs]
            
            # Calculate improvement
            initial_loss = losses[0]
            final_loss = losses[-1]
            total_improvement = initial_loss - final_loss
            
            if total_improvement <= 0:
                # No improvement or loss got worse, skip
                continue
                
            # Find epoch where model reached threshold% of total improvement
            convergence_epoch = None
            for i, loss in enumerate(losses):
                improvement = initial_loss - loss
                if improvement >= convergence_threshold * total_improvement:
                    convergence_epoch = epochs[i]
                    break
            
            if convergence_epoch is not None:
                # Get the training time per epoch
                epoch_times = []
                for key, value in run.data.metrics.items():
                    if key.startswith("epoch_time_"):
                        try:
                            epoch = int(key.split("_")[-1])
                            epoch_times.append(value)
                        except (ValueError, IndexError):
                            continue
                
                # Calculate average time per epoch
                avg_epoch_time = np.mean(epoch_times) if epoch_times else None
                
                # Store data
                convergence_data[model_name] = {
                    "epochs_to_converge": convergence_epoch,
                    "time_to_converge": convergence_epoch * avg_epoch_time if avg_epoch_time else None,
                    "final_loss": final_loss,
                    "initial_loss": initial_loss,
                    "improvement": total_improvement,
                    "convergence_threshold": convergence_threshold
                }
            
        except Exception as e:
            logger.error(f"Error processing run {run_id} for convergence plot: {e}")
    
    if not convergence_data:
        logger.warning("No convergence data available for plotting")
        return None
    
    # Create dataframe
    rows = []
    for model, data in convergence_data.items():
        rows.append({
            "Model": model,
            "Epochs to Converge": data["epochs_to_converge"],
            "Time to Converge (s)": data["time_to_converge"],
            "Final Loss": data["final_loss"],
            "Improvement %": 100 * (data["initial_loss"] - data["final_loss"]) / data["initial_loss"]
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by time to converge
    df_sorted = df.sort_values("Time to Converge (s)") if "Time to Converge (s)" in df.columns else df.sort_values("Epochs to Converge")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot bars for epochs to converge
    ax1 = plt.subplot(111)
    bars = ax1.bar(
        df_sorted["Model"],
        df_sorted["Epochs to Converge"],
        alpha=0.7,
        color="skyblue",
        edgecolor="black"
    )
    
    # Add time to converge as text annotations
    if "Time to Converge (s)" in df_sorted.columns:
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            if pd.notna(row["Time to Converge (s)"]):
                ax1.text(
                    i, 
                    row["Epochs to Converge"] + 0.5, 
                    f"{row['Time to Converge (s)']:.1f}s",
                    ha='center', 
                    va='bottom'
                )
    
    # Add final loss as a scatter plot on secondary axis
    if "Final Loss" in df_sorted.columns:
        ax2 = ax1.twinx()
        ax2.scatter(
            df_sorted["Model"],
            df_sorted["Final Loss"],
            color="red",
            s=100,
            marker="*",
            label="Final Loss"
        )
        ax2.set_ylabel("Final Loss Value", color="red", fontsize=12)
        
    # Add labels and title
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Epochs to Converge", fontsize=12)
    plt.title(f"Model Convergence Speed - {experiment_name}", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    
    # Add note about threshold
    plt.figtext(
        0.5, 0.01, 
        f"Note: Convergence defined as reaching {convergence_threshold*100}% of total loss improvement",
        ha="center", fontsize=10
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save plot
    plot_path = f"{experiment_dir}/convergence_speed_{timestamp}.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    return plot_path


def create_resource_efficiency_radar(run_ids, experiment_name, experiment_dir=None, timestamp=None):
    """
    Create a radar chart comparing multiple efficiency metrics across models.
    
    Args:
        run_ids: List of MLflow run IDs
        experiment_name: Name of the experiment for titling
        experiment_dir: Directory to save the plot in
        timestamp: Timestamp for consistent file naming
        
    Returns:
        Path to the generated plot
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.transforms import Affine2D
    except ImportError:
        logger.error("Required matplotlib components not available for radar chart")
        return None
    
    client = mlflow.tracking.MlflowClient()
    
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    # Create experiment directory if not provided
    if experiment_dir is None:
        experiment_dir = f"plots/efficiency_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
    
    # Define the metrics to include in the radar chart
    metrics = [
        "Training Speed", 
        "Accuracy", 
        "Parameter Efficiency",
        "Convergence Speed",
        "Memory Usage"
    ]
    
    # Function to create radar chart
    def radar_factory(num_vars, frame='circle'):
        """Create a radar chart with `num_vars` axes."""
        # Calculate evenly-spaced axis angles
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        
        # Rotate theta such that the first axis is at the top
        theta += np.pi/2
        
        def unit_poly_verts(theta):
            """Return vertices of polygon for subplot axes."""
            x0, y0, r = [0.5] * 3
            verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
            return verts
        
        class RadarAxes(plt.PolarAxes):
            """Class for creating a radar chart with matplotlib."""
            name = 'radar'
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')
            
            def fill(self, *args, **kwargs):
                """Override fill to draw a polygon."""
                return super().fill(*args, **kwargs)
            
            def plot(self, *args, **kwargs):
                """Override plot to draw a line."""
                return super().plot(*args, **kwargs)
            
            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)
            
            def _gen_axes_patch(self):
                # The Axes patch draws the spines
                if frame == 'circle':
                    return plt.Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return plt.Polygon(unit_poly_verts(theta), closed=True)
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)
            
            def draw(self, renderer):
                """Override draw method to handle the custom spines."""
                if frame == 'circle':
                    patch = plt.Circle((0.5, 0.5), 0.5)
                    patch.set_transform(self.transAxes)
                    patch.set_clip_on(False)
                    patch.set_fill(False)
                    self.add_patch(patch)
                    # Set grid line style if available
                    if hasattr(self, 'gridlines'):
                        self.gridlines.set_linestyle('-')
                elif frame == 'polygon':
                    patch = plt.Polygon(unit_poly_verts(theta), closed=True)
                    patch.set_transform(self.transAxes)
                    patch.set_clip_on(False)
                    patch.set_fill(False)
                    self.add_patch(patch)
                    # Set grid line style if available
                    if hasattr(self, 'gridlines'):
                        self.gridlines.set_linestyle('-')
                super().draw(renderer)
                
        # Register the custom projection
        from matplotlib.projections import register_projection
        register_projection(RadarAxes)
        return theta
        
    # Extract data for all models
    model_data = {}
    
    for run_id in run_ids:
        if not run_id:
            continue
            
        try:
            run = client.get_run(run_id)
            model_name = run.data.params.get("model", "Unknown")
            
            # Extract metrics
            training_time = run.data.metrics.get("training_time_seconds", np.nan)
            test_rmse = run.data.metrics.get("test_rmse", np.nan)
            param_count = run.data.metrics.get("param_count", np.nan)
            
            # If we don't have the basic metrics, skip this model
            if np.isnan(training_time) or np.isnan(test_rmse):
                continue
                
            # Get epochs to converge data
            epoch_times = []
            val_losses = {}
            for key, value in run.data.metrics.items():
                if key.startswith("epoch_time_"):
                    epoch_times.append(value)
                elif key.startswith("val_loss_epoch_"):
                    try:
                        epoch = int(key.split("_")[-1])
                        val_losses[epoch] = value
                    except (ValueError, IndexError):
                        continue
            
            # Calculate convergence speed if possible
            convergence_speed = np.nan
            if val_losses and len(val_losses) > 5:
                # Sort by epoch
                epochs = sorted(val_losses.keys())
                losses = [val_losses[e] for e in epochs]
                
                # Calculate improvement
                initial_loss = losses[0]
                final_loss = losses[-1]
                
                # Simple metric - average loss improvement per epoch
                if epochs[-1] > 0:
                    convergence_speed = (initial_loss - final_loss) / epochs[-1]
            
            # Get memory usage if available
            memory_usage = run.data.metrics.get("peak_memory_mb", np.nan)
            
            # Store data
            model_data[model_name] = {
                "Training Speed": training_time,
                "Accuracy": test_rmse,
                "Parameter Efficiency": param_count if not np.isnan(param_count) else np.nan,
                "Convergence Speed": convergence_speed,
                "Memory Usage": memory_usage
            }
            
        except Exception as e:
            logger.error(f"Error processing run {run_id} for radar chart: {e}")
    
    if not model_data:
        logger.warning("No data available for resource efficiency radar chart")
        return None
    
    # Normalize the data for radar chart (0 to 1 scale, where 1 is best)
    normalized_data = {}
    
    for metric in metrics:
        # Get all values for this metric
        values = [data[metric] for data in model_data.values() if not np.isnan(data[metric])]
        
        if not values:
            continue
            
        # Get min and max for normalization
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            continue
            
        # For each model, normalize this metric
        for model in model_data:
            if metric not in normalized_data:
                normalized_data[model] = {}
                
            value = model_data[model][metric]
            
            if np.isnan(value):
                normalized_data[model][metric] = 0.5  # Middle value for missing data
            else:
                # Different metrics have different directions (higher/lower is better)
                if metric in ["Accuracy", "Training Speed", "Memory Usage"]:
                    # Lower is better (invert normalization)
                    normalized_data[model][metric] = 1 - (value - min_val) / (max_val - min_val)
                else:
                    # Higher is better
                    normalized_data[model][metric] = (value - min_val) / (max_val - min_val)
    
    # Create a dataframe for plotting
    radar_data = []
    
    for model in normalized_data:
        model_values = []
        for metric in metrics:
            model_values.append(normalized_data[model].get(metric, 0))
        radar_data.append((model, model_values))
    
    # Can't create radar chart with insufficient data
    if not radar_data or len(radar_data[0][1]) < 3:
        logger.warning("Insufficient metrics for radar chart")
        return None
    
    # Create the radar chart
    theta = radar_factory(len(metrics), frame='polygon')
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    
    # Different line styles for each model
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
    
    # Plot each model
    for i, (model, values) in enumerate(radar_data):
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        
        ax.plot(theta, values, linestyle=line_style, marker=marker, label=model)
        ax.fill(theta, values, alpha=0.1)
    
    # Set labels and title
    ax.set_varlabels(metrics)
    plt.title(f"Model Efficiency Comparison - {experiment_name}", size=16, y=1.05)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add notes about metrics
    plt.figtext(0.5, 0.01, 
                "Note: All metrics normalized to 0-1 scale, with 1 being best performance.", 
                ha='center', fontsize=10)
    
    # Save plot
    plot_path = f"{experiment_dir}/efficiency_radar_{timestamp}.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    return plot_path


def create_model_scaling_comparison(run_ids, parameter_name="hidden_dim", experiment_dir=None, timestamp=None, dataset_type=None, data_version=None):
    """
    Create a visualization of how model performance scales with a specific parameter.
    
    Args:
        run_ids: List of MLflow run IDs
        parameter_name: Name of parameter to analyze scaling for
        experiment_dir: Directory to save the plot in
        timestamp: Timestamp for consistent file naming
        dataset_type: Optional filter for specific dataset type
        data_version: Optional filter for specific data version
        
    Returns:
        Path to the generated plot
    """
    client = mlflow.tracking.MlflowClient()
    
    # Extract data
    scaling_data = []
    
    for run_id in run_ids:
        if not run_id:
            continue
            
        try:
            run = client.get_run(run_id)
            
            # Get basic run info
            model_name = run.data.params.get("model", "Unknown")
            run_dataset_type = run.data.params.get("dataset_type", "Unknown")
            run_data_version = run.data.params.get("data_version", "Unknown")
            
            # Skip if dataset_type or data_version filters are applied and don't match
            if dataset_type and run_dataset_type != dataset_type:
                continue
            if data_version and run_data_version != data_version:
                continue
            
            # Get parameter value
            param_value = run.data.params.get(parameter_name, None)
            if param_value is None:
                continue
                
            # Try to convert to float
            try:
                param_value = float(param_value)
            except ValueError:
                continue
                
            # Get performance metrics
            training_time = run.data.metrics.get("training_time_seconds", np.nan)
            test_rmse = run.data.metrics.get("test_rmse", np.nan)
            test_r2 = run.data.metrics.get("test_r2", np.nan)
            
            # Add to data
            scaling_data.append({
                "Model": model_name,
                "Dataset": run_dataset_type,
                "Version": run_data_version,
                "Parameter": parameter_name,
                "Value": param_value,
                "Training Time": training_time,
                "RMSE": test_rmse,
                "R²": test_r2
            })
            
        except Exception as e:
            logger.error(f"Error processing run {run_id}: {e}")
    
    # Create dataframe
    if not scaling_data:
        logger.warning(f"No scaling data found for parameter {parameter_name}")
        return None
        
    df = pd.DataFrame(scaling_data)
    
    # Add dataset/version suffix for filtering
    dataset_suffix = f"_{dataset_type}" if dataset_type else ""
    version_suffix = f"_{data_version}" if data_version else ""
    filter_text = f" for {dataset_type}" if dataset_type else ""
    if data_version:
        filter_text += f" ({data_version})" if dataset_type else f" for {data_version}"
    
    # Create plot with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create a consistent color palette for models
    unique_models = df["Model"].unique()
    color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))
    model_to_color = {model: color_map[i] for i, model in enumerate(unique_models)}
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
    
    # Plot 1: Parameter vs. Training Time
    for i, model in enumerate(unique_models):
        model_data = df[df["Model"] == model]
        marker = markers[i % len(markers)]
        ax1.plot(
            model_data["Value"],
            model_data["Training Time"],
            marker=marker,
            linestyle='-',
            linewidth=2,
            markersize=8,
            label=model,
            color=model_to_color[model]
        )
    
    ax1.set_title(f"Training Time vs. {parameter_name.title()}{filter_text}", fontsize=14)
    ax1.set_xlabel(parameter_name.title(), fontsize=12)
    ax1.set_ylabel("Training Time (seconds)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    # Plot 2: Parameter vs. RMSE
    for i, model in enumerate(unique_models):
        model_data = df[df["Model"] == model]
        marker = markers[i % len(markers)]
        ax2.plot(
            model_data["Value"],
            model_data["RMSE"],
            marker=marker,
            linestyle='-',
            linewidth=2,
            markersize=8,
            label=model,
            color=model_to_color[model]
        )
    
    ax2.set_title(f"RMSE vs. {parameter_name.title()}{filter_text}", fontsize=14)
    ax2.set_xlabel(parameter_name.title(), fontsize=12)
    ax2.set_ylabel("RMSE (lower is better)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.3)
    
    # Add legend to figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), 
               ncol=min(len(df["Model"].unique()), 4), fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle(f"Model Scaling with {parameter_name.title()}{filter_text}", fontsize=16, y=0.98)
    
    # Use the same experiment directory as other functions for consistency
    if not os.path.exists(experiment_dir):
        experiment_dir = f"plots/efficiency_{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(experiment_dir, exist_ok=True)
    
    # Save plot to experiment directory with consistent timestamp
    plot_path = f"{experiment_dir}/scaling_{parameter_name}{dataset_suffix}{version_suffix}_{timestamp}.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    # Individual MLflow logging is now handled by the parent function
    # to ensure all plots are logged together
    
    return plot_path


def create_all_efficiency_visualizations(run_ids, experiment_name):
    """
    Create a comprehensive set of efficiency visualizations for a set of runs.
    
    Args:
        run_ids: List of MLflow run IDs
        experiment_name: Name of the experiment
        
    Returns:
        List of paths to generated visualizations
    """
    all_plots = []
    
    # Create timestamp once for consistent file naming across all functions
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    # Create a dedicated directory for this batch of efficiency visualizations
    efficiency_dir = f"plots/efficiency_{timestamp}"
    os.makedirs(efficiency_dir, exist_ok=True)
    logger.info(f"Created efficiency visualization directory: {efficiency_dir}")
    
    # Extract unique dataset and data versions for per-dataset filtering
    client = mlflow.tracking.MlflowClient()
    dataset_versions = set()
    
    # Get unique dataset/version combinations
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            dataset_type = run.data.params.get("dataset_type", "Unknown")
            data_version = run.data.params.get("data_version", "Unknown")
            dataset_versions.add((dataset_type, data_version))
        except Exception:
            continue
    
    # Basic efficiency comparisons - pass in the directory and timestamp
    efficiency_plots = create_efficiency_comparison(run_ids, experiment_name, efficiency_dir, timestamp)
    all_plots.extend(efficiency_plots)
    
    # Training progress comparison with shared experiment directory and timestamp
    train_progress_plot = plot_training_progress_comparison(
        run_ids, 
        "train_loss", 
        smoothing=0.2, 
        experiment_dir=efficiency_dir,
        timestamp=timestamp
    )
    if train_progress_plot:
        all_plots.append(train_progress_plot)
    
    # Scaling analysis (hidden_dim) with shared experiment directory and timestamp
    # First, overall plot with all runs
    scaling_plot = create_model_scaling_comparison(
        run_ids, 
        "hidden_dim", 
        experiment_dir=efficiency_dir,
        timestamp=timestamp
    )
    if scaling_plot:
        all_plots.append(scaling_plot)
    
    # Then, one plot per dataset/version combination
    for dataset_type, data_version in dataset_versions:
        dataset_scaling_plot = create_model_scaling_comparison(
            run_ids, 
            "hidden_dim", 
            experiment_dir=efficiency_dir,
            timestamp=timestamp,
            dataset_type=dataset_type,
            data_version=data_version
        )
        if dataset_scaling_plot:
            all_plots.append(dataset_scaling_plot)
    
    # Create accuracy vs. model complexity plot
    accuracy_complexity_plot = create_accuracy_complexity_plot(
        run_ids,
        experiment_name,
        efficiency_dir,
        timestamp
    )
    if accuracy_complexity_plot:
        all_plots.append(accuracy_complexity_plot)
        
    # Create convergence speed visualization
    convergence_plot = create_convergence_speed_plot(
        run_ids,
        experiment_name,
        efficiency_dir,
        timestamp
    )
    if convergence_plot:
        all_plots.append(convergence_plot)
    
    # Create resource efficiency radar chart
    radar_plot = create_resource_efficiency_radar(
        run_ids,
        experiment_name,
        efficiency_dir,
        timestamp
    )
    if radar_plot:
        all_plots.append(radar_plot)
    
    # Logging is now handled by the runner.py to ensure consistent artifact paths
    
    return all_plots