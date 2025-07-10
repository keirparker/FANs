#!/usr/bin/env python
"""
Visualization utilities for ML model training and evaluation.

This module provides high-quality visualization capabilities for machine learning
model training history, predictions, and integration with MLflow.

Author: GitHub Copilot for keirparker
Last updated: 2025-05-08
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Callable
from loguru import logger
from datetime import datetime
import mlflow
import ray
import shutil

def init_ray_for_mac(num_cpus=None):
    """Initialize Ray with settings optimized for M2 Macbook.
    
    Args:
        num_cpus: Number of CPUs to use. If None, uses all available minus 2.
        
    Returns:
        bool: True if Ray was successfully initialized, False otherwise
    """
    if ray.is_initialized():
        return True

    import multiprocessing
    
    if num_cpus is None:
        # Leave 2 CPUs free for system operations on M2 Mac
        available_cpus = multiprocessing.cpu_count()
        num_cpus = max(1, available_cpus - 2)
    
    try:
        ray.init(
            num_cpus=num_cpus,
            include_dashboard=False,
            ignore_reinit_error=True,
            _temp_dir="/tmp/ray_temp",  # Prevent permissions issues on macOS
            _system_config={
                # MacOS-specific configurations
                "worker_register_timeout_seconds": 60,
                "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}}',
                "max_io_workers": 4,  # Reduce I/O worker threads for Mac
                "object_store_full_delay_ms": 100  # More aggressive memory management
            },
            logging_level=logging.WARNING
        )
        logger.info(f"Ray initialized with {num_cpus} CPUs")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize Ray: {e}. Will continue without parallel processing.")
        return False

# Configure matplotlib for publication-quality output
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 600, "axes.grid": True,
    "grid.linestyle": ":", "grid.alpha": 0.3, 
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.markersize": 0, "lines.linewidth": 1.5, 
    "xtick.direction": 'in', "ytick.direction": 'in',
    "axes.titlesize": 14, "axes.labelsize": 12
})

# Colorblind-friendly palette
COLOR_PALETTE = sns.color_palette("colorblind", 8)


class VisualizationConfig:
    """Configuration for visualization settings."""

    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize visualization configuration."""
        default_config = {
            "theme": "light",  # 'light', 'dark', 'publication'
            "interactive": False,  # Generate interactive plots
            "width": 10,  # Default figure width in inches
            "height": 6,  # Default figure height in inches
            "dpi": 300,  # DPI for raster formats
            "title_size": 14,  # Title font size
            "label_size": 12,  # Label font size
        }

        # Apply user config over defaults
        self.config = default_config
        if config_dict:
            self.config.update(config_dict)

    def get_figure_kwargs(self) -> Dict:
        """Get kwargs for figure creation."""
        return {
            "figsize": (self.config["width"], self.config["height"]),
            "dpi": self.config["dpi"],
        }

    def style_axis(self, ax):
        """Apply styling to an axis object."""
        # Configure grid
        ax.grid(True, alpha=0.6, linestyle="--")

        # Configure spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # Configure font sizes
        ax.title.set_fontsize(self.config["title_size"])
        ax.xaxis.label.set_fontsize(self.config["label_size"])
        ax.yaxis.label.set_fontsize(self.config["label_size"])


def _generate_plot_filename(model_name, dataset_type, data_version, plot_type, experiment_name=None, run_number=None):
    """Generate unique filename for a plot."""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    # Get time_series base directory for proper pathing
    time_series_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(time_series_dir, "plots")
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # If experiment name and run number provided, create directory structure
    if experiment_name and run_number:
        # Clean experiment name for safe path usage
        safe_exp_name = experiment_name.replace(" ", "_").replace("/", "_")
        
        # Create experiment directory
        exp_dir = os.path.join(plots_dir, safe_exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create run directory
        run_dir = os.path.join(exp_dir, f"run_{run_number}")
        os.makedirs(run_dir, exist_ok=True)
        
        return os.path.join(run_dir, f"{model_name}_{dataset_type}_{data_version}_{plot_type}.png")
    else:
        # Flat structure with timestamp for backward compatibility
        return os.path.join(plots_dir, f"{model_name}_{dataset_type}_{data_version}_{plot_type}_{timestr}.png")


def plot_training_history(
    history: Dict,
    model_name: str,
    dataset_type: str,
    data_version: str,
    viz_config: Optional[VisualizationConfig] = None,
    experiment_name: Optional[str] = None,
    run_number: Optional[int] = None,
) -> List[str]:
    """
    Generate visualizations of model training history.

    Args:
        history: Dictionary containing training metrics
        model_name: Name of the model
        dataset_type: Type of dataset used
        data_version: Version of data
        viz_config: Visualization configuration

    Returns:
        List of saved plot file paths
    """
    # Create visualization config if not provided
    if viz_config is None:
        viz_config = VisualizationConfig()

    # Create list to store plot filenames
    plot_files = []

    # Create a composite figure for the training history
    fig = plt.figure(**viz_config.get_figure_kwargs())
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])

    # Plot 1: Loss curves (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    # Generate epochs based on the actual number of training loss entries
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax1.plot(
        epochs,
        history["train_loss"],
        label="Training",
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=2,
        color=COLOR_PALETTE[0],
    )

    if "val_loss" in history and history["val_loss"] is not None:
        # Make sure the epochs match validation loss length
        val_epochs = list(range(1, len(history["val_loss"]) + 1))
        ax1.plot(
            val_epochs,
            history["val_loss"],
            label="Validation",
            marker="x",
            markersize=4,
            linestyle="--",
            linewidth=2,
            color=COLOR_PALETTE[1],
        )

    # Add annotations for min loss points
    min_train_idx = np.argmin(history["train_loss"])
    min_train_loss = history["train_loss"][min_train_idx]
    ax1.annotate(
        f"{min_train_loss:.4f}",
        xy=(min_train_idx + 1, min_train_loss),
        xytext=(5, -15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )

    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    viz_config.style_axis(ax1)

    # Plot 2: Learning Rate vs Epochs (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if "learning_rate" in history:
        # Make sure epochs match learning rate length
        lr_epochs = list(range(1, len(history["learning_rate"]) + 1))
        ax2.plot(
            lr_epochs,
            history["learning_rate"],
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=2,
            color=COLOR_PALETTE[2],
        )
        ax2.set_title("Learning Rate Schedule")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")
    else:
        ax2.text(
            0.5,
            0.5,
            "Learning rate data not available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
    viz_config.style_axis(ax2)

    # Plot 3: R² Score (middle row)
    ax3 = fig.add_subplot(gs[1, :])

    if "metrics" in history and history["metrics"] and "r2" in history["metrics"][0]:
        r2_values = [metrics["r2"] for metrics in history["metrics"]]
        # Make sure epochs match metrics length
        metrics_epochs = list(range(1, len(r2_values) + 1))
        ax3.plot(
            metrics_epochs,
            r2_values,
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=2,
            color=COLOR_PALETTE[3],
        )

        # Add performance bands
        ax3.axhspan(0.9, 1.0, alpha=0.1, color="green", label="Excellent")
        ax3.axhspan(0.7, 0.9, alpha=0.1, color="lightgreen", label="Good")
        ax3.axhspan(0.5, 0.7, alpha=0.1, color="yellow", label="Moderate")
        ax3.axhspan(0, 0.5, alpha=0.1, color="red", label="Poor")

        ax3.set_title("R² Score Evolution")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("R² Score")
        ax3.legend(loc="lower right")
    else:
        ax3.text(
            0.5,
            0.5,
            "R² metric not available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax3.transAxes,
            fontsize=12,
        )
    viz_config.style_axis(ax3)

    # Plot 4: Error metrics (bottom row)
    ax4 = fig.add_subplot(gs[2, :])

    if "metrics" in history and history["metrics"]:
        has_metrics = False

        # Check for RMSE
        if all("rmse" in m for m in history["metrics"]):
            rmse_values = [metrics["rmse"] for metrics in history["metrics"]]
            # Make sure epochs match metrics length
            metrics_epochs = list(range(1, len(rmse_values) + 1))
            ax4.plot(
                metrics_epochs,
                rmse_values,
                marker="o",
                markersize=4,
                linestyle="-",
                linewidth=2,
                color=COLOR_PALETTE[4],
                label="RMSE",
            )
            has_metrics = True

        # Check for MAE
        if all("mae" in m for m in history["metrics"]):
            mae_values = [metrics["mae"] for metrics in history["metrics"]]
            # Make sure epochs match metrics length
            mae_epochs = list(range(1, len(mae_values) + 1))
            ax4.plot(
                mae_epochs,
                mae_values,
                marker="^",
                markersize=4,
                linestyle="--",
                linewidth=2,
                color=COLOR_PALETTE[5],
                label="MAE",
            )
            has_metrics = True

        if has_metrics:
            ax4.set_title("Error Metrics Evolution")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Error")
            ax4.legend(loc="upper right")
        else:
            ax4.text(
                0.5,
                0.5,
                "Error metrics not available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
    else:
        ax4.text(
            0.5,
            0.5,
            "Metrics data not available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax4.transAxes,
            fontsize=12,
        )
    viz_config.style_axis(ax4)

    # Add overall title
    fig.suptitle(
        f"{model_name} on {dataset_type} ({data_version})", fontsize=14, y=0.98
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    plot_path = _generate_plot_filename(
        model_name, dataset_type, data_version, "training_history", 
        experiment_name=experiment_name, run_number=run_number
    )
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    plot_files.append(plot_path)

    # Generate epoch timing plot if time data is available
    if "epoch_times" in history and len(history["epoch_times"]) > 0:
        fig = plt.figure(**viz_config.get_figure_kwargs())
        # Make sure epochs match epoch_times length
        time_epochs = list(range(1, len(history["epoch_times"]) + 1))
        plt.plot(
            time_epochs,
            history["epoch_times"],
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=2,
            color=COLOR_PALETTE[6],
        )

        # Add moving average line
        window_size = min(5, len(history["epoch_times"]))
        if window_size > 1:
            conv_filter = np.ones(window_size) / window_size
            smoothed = np.convolve(history["epoch_times"], conv_filter, mode="valid")
            valid_epochs = list(range(1, len(smoothed) + 1))
            plt.plot(
                valid_epochs,
                smoothed,
                linestyle="--",
                linewidth=2,
                color=COLOR_PALETTE[6],
                alpha=0.6,
                label="Moving Average",
            )

        plt.title("Epoch Processing Time")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add timing stats
        avg_time = np.mean(history["epoch_times"])
        total_time = np.sum(history["epoch_times"])
        stats_text = f"Avg: {avg_time:.2f}s | Total: {total_time:.0f}s"
        plt.annotate(
            stats_text,
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            horizontalalignment="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

        # Save plot
        timing_plot_path = _generate_plot_filename(
            model_name, dataset_type, data_version, "epoch_times",
            experiment_name=experiment_name, run_number=run_number
        )
        plt.savefig(timing_plot_path, bbox_inches="tight")
        plt.close(fig)
        plot_files.append(timing_plot_path)

    # Generate interactive HTML version if requested
    if viz_config.config["interactive"]:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create interactive figure
            fig = make_subplots(
                rows=3,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]],
                subplot_titles=(
                    "Loss Curves",
                    "Learning Rate",
                    "R² Score Evolution",
                    "Error Metrics Evolution",
                ),
            )

            # Add loss curves
            epochs = list(range(1, len(history["train_loss"]) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["train_loss"],
                    mode="lines+markers",
                    name="Training Loss",
                    line=dict(color="rgb(31, 119, 180)"),
                ),
                row=1,
                col=1,
            )

            if "val_loss" in history and history["val_loss"] is not None:
                val_epochs = list(range(1, len(history["val_loss"]) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=val_epochs,
                        y=history["val_loss"],
                        mode="lines+markers",
                        name="Validation Loss",
                        line=dict(color="rgb(255, 99, 71)"),
                    ),
                    row=1,
                    col=1,
                )

            # Add interactive plot elements
            interactive_path = (
                _generate_plot_filename(
                    model_name,
                    dataset_type,
                    data_version,
                    "training_history_interactive",
                )[:-4]
                + ".html"
            )
            fig.write_html(interactive_path)
            plot_files.append(interactive_path)

        except ImportError:
            logger.warning("plotly not installed, skipping interactive plot generation")

    return plot_files


def plot_model_predictions(
    model_name: str,
    dataset_type: str,
    data_version: str,
    t_train: np.ndarray,
    data_train: np.ndarray,
    t_test: np.ndarray,
    data_test: np.ndarray,
    predictions: np.ndarray,
    true_func: Optional[Callable] = None,
    uncertainty: Optional[np.ndarray] = None,
    viz_config: Optional[VisualizationConfig] = None,
    experiment_name: Optional[str] = None,
    run_number: Optional[int] = None,
) -> str:
    """
    Generate visualization of model predictions with statistical analysis.
    
    The visualization includes:
    - Shaded training region (light green)
    - True function as a solid red line (if provided)
    - Model predictions as a solid line
    - Clear division between training and test regions
    - No scatter points for cleaner visualization
    - Reduced frequency for better visibility in -70 to +70 range
    
    Args:
        model_name: Name of the model
        dataset_type: Type of dataset
        data_version: Version of data
        t_train: Training time points
        data_train: Training data values
        t_test: Test time points
        data_test: Test data values
        predictions: Model predictions on test data
        true_func: True underlying function (optional)
        uncertainty: Model prediction uncertainty (optional)
        viz_config: Visualization configuration
        experiment_name: Name of the experiment (for file naming)
        run_number: Run number (for file naming)

    Returns:
        str: Path to the saved plot
    """
    # Create visualization config if not provided
    if viz_config is None:
        viz_config = VisualizationConfig()

    # Create figure with 2 rows
    fig = plt.figure(**viz_config.get_figure_kwargs())
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])

    # Main plot (predictions vs actual)
    ax1 = fig.add_subplot(gs[0, :])

    # Sort data points for smoother curve plotting
    sort_idx_test = np.argsort(t_test)
    t_test_sorted = t_test[sort_idx_test]
    data_test_sorted = data_test[sort_idx_test]
    predictions_sorted = predictions[sort_idx_test]

    # Find the min and max values for t_train and t_test
    t_min = min(min(t_train), min(t_test))
    t_max = max(max(t_train), max(t_test))
    
    # Shade the training region with light green
    train_min = min(t_train)
    train_max = max(t_train)
    ax1.axvspan(train_min, train_max, alpha=0.15, color='green', label="Training Region")
    
    # Plot the true function with extremely high sampling for perfect representation
    if true_func is not None:
        # Calculate appropriate number of points based on range
        # For sine functions, need enough points to capture high frequencies
        range_size = t_max - t_min
        # Use at least 100 points per unit for ultra-high quality representation
        # For wide ranges like -70 to +70, this can use hundreds of thousands of points
        num_points = max(10000, int(range_size * 100))
        
        # For sine wave specifically, ensure even higher density
        if "sin" in str(true_func):
            num_points = max(num_points, int(range_size * 200))
        
        # Log information about the high-fidelity rendering
        print(f"Generating true function with {num_points} points for high-fidelity visualization")
        
        # Generate the dense sampling
        t_dense = np.linspace(t_min, t_max, num_points)
        y_dense = true_func(t_dense)
        
        ax1.plot(
            t_dense,
            y_dense,
            linestyle="-",
            color="black",  # Changed from red to black as requested
            alpha=0.9,
            linewidth=1.5,  # Reduced line width from 2.0 to 1.5
            label="True Function",
        )
    
    # Note: We removed the training data line plot
    # Note: We removed the test data line plot

    # Plot model predictions as a continuous line
    ax1.plot(
        t_test_sorted,
        predictions_sorted,
        color=COLOR_PALETTE[3],
        linestyle="-",
        linewidth=1.5,  # Reduced from 2.5 to 1.5
        label="Model Prediction",
    )
    
    # Add a subtle vertical line to indicate division between train/test if not overlapping
    if train_max < max(t_test):
        ax1.axvline(x=train_max, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)  # Made more subtle

    # Add uncertainty bands if provided
    if uncertainty is not None:
        uncertainty_sorted = uncertainty[sort_idx_test]
        # Plot 95% confidence interval
        ax1.fill_between(
            t_test_sorted,
            predictions_sorted - 2 * uncertainty_sorted,
            predictions_sorted + 2 * uncertainty_sorted,
            alpha=0.2,
            color=COLOR_PALETTE[4],
            label="95% Confidence",
        )

    # Calculate residuals, handling any NaNs
    residuals = data_test - predictions
    
    # Replace any NaNs in residuals with zeros for visualization purposes
    if np.isnan(residuals).any():
        print(f"Warning: Found {np.sum(np.isnan(residuals))} NaN values in residuals. Replacing with zeros for visualization.")
        residuals = np.nan_to_num(residuals, nan=0.0)

    # Calculate metrics with NaN handling
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    # Create NaN-free versions of the data for metric calculation
    data_test_clean = np.copy(data_test)
    predictions_clean = np.copy(predictions)
    
    # Identify NaN positions in either array
    nan_mask = np.isnan(data_test_clean) | np.isnan(predictions_clean)
    if nan_mask.any():
        print(f"Warning: Removing {np.sum(nan_mask)} NaN values for metric calculation")
        data_test_clean = data_test_clean[~nan_mask]
        predictions_clean = predictions_clean[~nan_mask]
    
    # Calculate metrics on clean data
    try:
        mse = mean_squared_error(data_test_clean, predictions_clean)
        rmse = np.sqrt(mse)
        r2 = r2_score(data_test_clean, predictions_clean)
        mae = mean_absolute_error(data_test_clean, predictions_clean)
    except Exception as e:
        print(f"Error calculating metrics: {e}. Using fallback values.")
        mse = rmse = mae = 99.0
        r2 = -1.0

    # Format metrics text with proper notation
    metrics_text = f"RMSE = {rmse:.4f}\nMAE = {mae:.4f}\n$R^2$ = {r2:.4f}"

    # Create text box with metrics - more subtle academic style
    ax1.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='lightgray'),
    )

    # Customize main plot - more academic labels with units
    ax1.set_title(f"{model_name} Model Performance")
    ax1.set_xlabel("Input Variable ($x$)")
    ax1.set_ylabel("Response Variable ($y$)")
    
    # Add text indicators for the regions
    mid_train = (train_min + train_max) / 2
    ax1.text(mid_train, ax1.get_ylim()[1] * 0.9, "Training Region", 
             ha='center', va='top', alpha=0.7, fontsize=9, color='green')
    
    # Add test region label if different from training
    if train_max < max(t_test):
        mid_test = (train_max + max(t_test)) / 2
        ax1.text(mid_test, ax1.get_ylim()[1] * 0.9, "Test Region", 
                ha='center', va='top', alpha=0.7, fontsize=9, color='purple')
    
    ax1.legend(loc="upper right", framealpha=0.9, edgecolor='lightgray')
    viz_config.style_axis(ax1)

    # Residual plot (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])

    # Plot residuals as a line for academic style
    # Sort residuals by input for a clean line
    sort_idx_res = np.argsort(t_test)
    t_test_res_sorted = t_test[sort_idx_res]
    residuals_sorted = residuals[sort_idx_res]
    
    ax2.plot(
        t_test_res_sorted,
        residuals_sorted,
        color=COLOR_PALETTE[4],
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add zero reference line
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.7)

    ax2.set_title("Residual Analysis")
    ax2.set_xlabel("Input Variable ($x$)")
    ax2.set_ylabel("Residual ($y - \hat{y}$)")
    viz_config.style_axis(ax2)

    # Residual histogram (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])

    # Create histogram with KDE (academic style)
    sns.histplot(
        residuals, 
        bins=15, 
        kde=True, 
        ax=ax3, 
        color=COLOR_PALETTE[5], 
        alpha=0.7,
        line_kws={'linewidth': 2, 'color': COLOR_PALETTE[6]}
    )
    
    # Remove top and right spines for cleaner academic look
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Add vertical line at zero
    ax3.axvline(x=0, color="gray", linestyle="--", alpha=0.7)

    # Add statistical summary
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    stats_text = f"μ = {mean_res:.4f}\nσ = {std_res:.4f}"

    ax3.text(
        0.95,
        0.95,
        stats_text,
        transform=ax3.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax3.set_title("Error Distribution")
    ax3.set_xlabel("Residual Value ($y - \hat{y}$)")
    ax3.set_ylabel("Frequency")
    viz_config.style_axis(ax3)

    # Add overall title with more academic formatting
    fig.suptitle(
        f"{model_name}: {dataset_type} Dataset Performance", 
        fontsize=16, 
        y=0.98,
        fontweight='normal',
        fontfamily='serif'
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    plot_path = _generate_plot_filename(
        model_name, dataset_type, data_version, "predictions",
        experiment_name=experiment_name, run_number=run_number
    )
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    # Generate interactive version if requested
    if viz_config.config["interactive"]:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create interactive figure
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[[{"colspan": 2}, None], [{}, {}]],
                subplot_titles=(
                    "Model Predictions",
                    "Residuals",
                    "Residual Distribution",
                ),
            )

            # Add interactive elements
            interactive_path = (
                _generate_plot_filename(
                    model_name, dataset_type, data_version, "predictions_interactive"
                )[:-4]
                + ".html"
            )

            # Save basic interactive plot
            fig.write_html(interactive_path)

        except ImportError:
            logger.warning("plotly not installed, skipping interactive plot generation")

    return plot_path

def log_plots_to_mlflow(
    plot_paths: List[str],
    run_id: str = None,
    artifact_path: str = "plots",
    timestamp: bool = True,
) -> None:
    """
    Log generated plots to MLflow.

    Args:
        plot_paths: List of paths to plots
        run_id: MLflow run ID (uses active run if None)
        artifact_path: Path within MLflow where artifacts will be stored
        timestamp: Whether to add timestamp to artifact path
    """
    try:
        import mlflow

        # Add timestamp to artifact path if requested
        if timestamp:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            artifact_path = f"{artifact_path}/{current_time}"

        # Check if we have an active run or need to use the provided run_id
        active_run = mlflow.active_run()

        if active_run is not None and run_id is None:
            # Use the active run - no need to create a new run context
            for plot_path in plot_paths:
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, artifact_path)
                    logger.info(f"Logged plot {plot_path} to MLflow")
                else:
                    logger.warning(f"Plot file not found: {plot_path}")
        else:
            # Use the provided run_id or create a new run if neither is available
            with mlflow.start_run(run_id=run_id, nested=True):
                for plot_path in plot_paths:
                    if os.path.exists(plot_path):
                        mlflow.log_artifact(plot_path, artifact_path)
                        logger.info(f"Logged plot {plot_path} to MLflow")
                    else:
                        logger.warning(f"Plot file not found: {plot_path}")

        # Log success message with user info
        logger.info(
            f"All plots successfully logged to MLflow by {os.environ.get('USER', 'keirparker')} "
            f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    except ImportError:
        logger.warning("MLflow not installed, skipping logging of plots")
    except Exception as e:
        logger.error(f"Error logging plots to MLflow: {e}")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd

# Set styling for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)


def plot_separate_loss_curves(
    run_ids: List[str],
    experiment_name: str = None,
    group_by: str = "dataset_type",
    output_dir: str = None,
    show_markers: bool = True,
    line_smoothing: float = 0.0,
    max_per_plot: int = 4,
):
    """
    Create separate loss plots grouped by dataset, model, or data version.
    
    Args:
        run_ids: List of MLflow run IDs to include
        experiment_name: Name for the experiment/collection of plots
        group_by: How to group plots - 'dataset_type', 'data_version', or 'model'
        output_dir: Directory to save output plots (defaults to time_series/plots if None)
        show_markers: Whether to show data points as markers
        line_smoothing: Smoothing factor (0.0 = raw data, higher values = more smoothing)
        max_per_plot: Maximum number of curves to show in a single plot
        
    Returns:
        List of paths to saved plots
    """
    logger.info(f"Generating separated loss plots grouped by {group_by}")
    
    # Use time_series/plots if output_dir not specified
    if output_dir is None:
        time_series_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(time_series_dir, "plots")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get data from MLflow runs
    client = mlflow.tracking.MlflowClient()

    # Store metrics by group
    metrics_data = []
    groups = set()

    # Extract data from runs
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)

            # Extract metadata
            model = run.data.params.get("model", "unknown")
            dataset_type = run.data.params.get("dataset_type", "unknown")
            data_version = run.data.params.get("data_version", "unknown")

            # Extract epoch-specific metrics
            for key, value in run.data.metrics.items():
                if key.startswith("train_loss_epoch_") or key.startswith(
                    "val_loss_epoch_"
                ):
                    try:
                        parts = key.split("_")
                        metric_type = "train" if parts[0] == "train" else "val"
                        epoch = int(parts[-1])

                        # Determine grouping key
                        if group_by == "dataset_type":
                            group = dataset_type
                        elif group_by == "data_version":
                            group = data_version
                        elif group_by == "model":
                            group = model
                        else:
                            group = dataset_type  # Default to dataset

                        groups.add(group)

                        # Add to metrics data
                        metrics_data.append(
                            {
                                "group": group,
                                "model": model,
                                "dataset_type": dataset_type,
                                "data_version": data_version,
                                "metric_type": metric_type,
                                "epoch": epoch,
                                "value": value,
                                "run_id": run_id,
                                "label": f"{model} ({data_version}, {metric_type})",
                            }
                        )

                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            logger.warning(f"Error processing run {run_id}: {e}")

    if not metrics_data:
        logger.warning("No loss data found in the provided runs")
        return []

    # Convert to dataframe for easier manipulation
    df = pd.DataFrame(metrics_data)

    # Sort by epoch for each label
    df = df.sort_values(by=["group", "label", "epoch"])

    # Generate plots - one per group
    plot_paths = []

    for group in sorted(groups):
        # Get data for this group
        group_df = df[df["group"] == group]

        # Get unique model/version/type combinations
        unique_labels = group_df["label"].unique()

        # Create multiple plots if there are too many curves
        for plot_idx, labels_subset in enumerate(
            [
                unique_labels[i : i + max_per_plot]
                for i in range(0, len(unique_labels), max_per_plot)
            ]
        ):
            # Create figure
            plt.figure(figsize=(10, 6))

            # Color palette
            colors = sns.color_palette("husl", len(labels_subset))

            # Plot each curve
            legend_entries = []

            for i, label in enumerate(labels_subset):
                curve_data = group_df[group_df["label"] == label]

                if curve_data.empty:
                    continue

                epochs = curve_data["epoch"].values
                values = curve_data["value"].values

                # Apply smoothing if requested
                if line_smoothing > 0:
                    values = apply_smoothing(values, line_smoothing)

                # Determine line style based on metric type
                is_val = "val" in label
                linestyle = "--" if is_val else "-"
                marker = "x" if is_val else "o"
                alpha = 0.7 if is_val else 1.0

                # Plot the line
                line = plt.plot(
                    epochs,
                    values,
                    linestyle=linestyle,
                    color=colors[i],
                    linewidth=2,
                    alpha=alpha,
                    label=label,
                )[0]

                # Add markers if requested
                if show_markers:
                    plt.scatter(
                        epochs,
                        curve_data["value"].values,  # Original unsmoothed values
                        marker=marker,
                        s=30,
                        color=line.get_color(),
                        alpha=0.8,
                        edgecolors="white",
                        linewidth=0.5,
                    )

                legend_entries.append(label)

            # Customize plot
            plt.title(f"Loss Curves - {group}", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)

            # Add legend
            if legend_entries:
                plt.legend(fontsize=10, loc="best")

            # Determine if log scale would be better
            if len(legend_entries) > 0:
                values = group_df[group_df["label"].isin(labels_subset)]["value"]
                if values.max() / values.min() > 10:
                    plt.yscale("log")

            # Adjust layout
            plt.tight_layout()

            # Save plot
            timestr = time.strftime("%Y%m%d-%H%M%S")
            plot_filename = (
                f"{output_dir}/loss_plot_{group}_part{plot_idx + 1}_{timestr}.png"
            )
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            plot_paths.append(plot_filename)
            logger.info(f"Created loss plot: {plot_filename}")

    return plot_paths


def plot_loss_grid(
    run_ids: List[str],
    output_dir: str = None,
    show_markers: bool = True,
    line_smoothing: float = 0.0,
):
    """
    Create a grid of loss plots organized by dataset and data version.
    
    Args:
        run_ids: List of MLflow run IDs to include
        output_dir: Directory to save output plots (defaults to time_series/plots if None)
        show_markers: Whether to show data points as markers
        line_smoothing: Smoothing factor (0.0 = raw data, higher values = more smoothing)
        
    Returns:
        Path to the saved grid plot
    """
    logger.info("Generating loss plot grid")
    
    # Use time_series/plots if output_dir not specified
    if output_dir is None:
        time_series_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(time_series_dir, "plots")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get data from MLflow runs
    client = mlflow.tracking.MlflowClient()

    # Extract metadata from runs to determine grid dimensions
    datasets = set()
    data_versions = set()
    models = {}  # Dictionary of model names to data

    # Extract data from runs
    run_data = {}

    for run_id in run_ids:
        try:
            run = client.get_run(run_id)

            # Extract metadata
            model = run.data.params.get("model", "unknown")
            dataset_type = run.data.params.get("dataset_type", "unknown")
            data_version = run.data.params.get("data_version", "unknown")

            datasets.add(dataset_type)
            data_versions.add(data_version)

            key = f"{model}_{dataset_type}_{data_version}"
            run_data[key] = {"train": {}}

            if model not in models:
                models[model] = {"color": None}  # Color will be assigned later

            # Extract epoch-specific metrics
            for metric_key, value in run.data.metrics.items():
                if metric_key.startswith("train_loss_epoch_"):
                    try:
                        epoch = int(metric_key.split("_")[-1])
                        run_data[key]["train"][epoch] = value
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            logger.warning(f"Error processing run {run_id}: {e}")

    if not run_data:
        logger.warning("No loss data found in the provided runs")
        return None

    # Assign colors to models
    model_colors = dict(zip(models.keys(), sns.color_palette("husl", len(models))))
    for model in models:
        models[model]["color"] = model_colors[model]

    # Sort datasets and data versions for consistent ordering
    datasets = sorted(datasets)
    data_versions = sorted(data_versions)

    # Create grid of subplots
    num_rows = len(datasets)
    num_cols = len(data_versions)

    # Calculate figure size based on grid dimensions
    fig_width = max(3 * num_cols, 8)
    fig_height = max(3 * num_rows, 6)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Track legend handles and labels
    legend_handles = []
    legend_labels = []

    # Fill each subplot
    for i, dataset in enumerate(datasets):
        for j, data_version in enumerate(data_versions):
            ax = axes[i, j]

            # Set title for this subplot
            ax.set_title(f"{dataset} - {data_version}")

            # Add data for each model
            for model_name, model_info in models.items():
                key = f"{model_name}_{dataset}_{data_version}"

                if key not in run_data:
                    continue

                # Process train loss
                if run_data[key]["train"]:
                    epochs = sorted(run_data[key]["train"].keys())
                    values = [run_data[key]["train"][e] for e in epochs]

                    # Apply smoothing if requested
                    if line_smoothing > 0 and len(values) > 3:
                        smooth_values = apply_smoothing(values, line_smoothing)
                    else:
                        smooth_values = values

                    # Plot line
                    line = ax.plot(
                        epochs,
                        smooth_values,
                        linestyle="-",
                        color=model_info["color"],
                        linewidth=2,
                        label=f"{model_name}",
                    )[0]

                    # Add markers if requested
                    if show_markers:
                        ax.scatter(
                            epochs,
                            values,  # Original unsmoothed values
                            marker="o",
                            s=20,
                            color=model_info["color"],
                            alpha=0.7,
                            edgecolors="white",
                            linewidth=0.5,
                        )

                    # Only add to legend once
                    if i == 0 and j == 0:
                        legend_handles.append(line)
                        legend_labels.append(f"{model_name} (train)")

                # Process val loss
                if run_data[key]["val"]:
                    epochs = sorted(run_data[key]["val"].keys())
                    values = [run_data[key]["val"][e] for e in epochs]

                    # Apply smoothing if requested
                    if line_smoothing > 0 and len(values) > 3:
                        smooth_values = apply_smoothing(values, line_smoothing)
                    else:
                        smooth_values = values

                    # Plot line with dashed style for validation
                    val_line = ax.plot(
                        epochs,
                        smooth_values,
                        linestyle="--",
                        color=model_info["color"],
                        linewidth=2,
                        alpha=0.7,
                        label=f"{model_name} (val)",
                    )[0]

                    # Add markers if requested
                    if show_markers:
                        ax.scatter(
                            epochs,
                            values,  # Original unsmoothed values
                            marker="x",
                            s=20,
                            color=model_info["color"],
                            alpha=0.7,
                        )

                    # Only add to legend once
                    if i == 0 and j == 0:
                        legend_handles.append(val_line)
                        legend_labels.append(f"{model_name} (val)")

            # Only label axes for edge subplots
            if j == 0:
                ax.set_ylabel("Loss")
            if i == num_rows - 1:
                ax.set_xlabel("Epoch")

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.5)

    # Add overall title
    fig.suptitle("Loss Curves Comparison", fontsize=16)

    # Add single legend for the entire figure
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=min(len(legend_handles), 4),
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save figure
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plot_filename = f"{output_dir}/loss_grid_{timestr}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created loss grid plot: {plot_filename}")
    return plot_filename


def apply_smoothing(values: List[float], smoothing_factor: float) -> np.ndarray:
    """Apply moving average smoothing to a list of values."""
    if smoothing_factor <= 0 or len(values) < 4:
        return values

    values_array = np.array(values)
    smoothing_factor = min(0.9, max(0, smoothing_factor))
    window_size = max(3, int(len(values) * smoothing_factor))
    
    if window_size % 2 == 0:
        window_size += 1

    window = np.ones(window_size) / window_size
    smoothed = np.convolve(values_array, window, mode="same")
    
    half_window = window_size // 2

    # Fix edges
    for i in range(half_window):
        window_size_edge = i + half_window + 1
        if window_size_edge > 0:
            smoothed[i] = np.sum(values_array[:window_size_edge]) / window_size_edge
            
            idx = len(values) - i - 1
            smoothed[idx] = np.sum(values_array[-window_size_edge:]) / window_size_edge

    return smoothed


def log_enhanced_plots_to_mlflow(
    experiment_id: str,
    run_ids: List[str],
    output_dir: str = None,
    artifact_path: str = "enhanced_plots",
    run_number: int = None,
    use_ray: bool = True,
    num_cpus: Optional[int] = None,
):
    """
    Generate and log enhanced loss plots to MLflow.
    
    This function is deprecated - use create_enhanced_visualizations instead.

    Args:
        experiment_id: MLflow experiment ID
        run_ids: List of run IDs to include in visualization
        output_dir: Directory to save intermediate plots (defaults to time_series/plots if None)
        artifact_path: Path within MLflow artifacts to store plots
        run_number: Run number identifier (optional)
        use_ray: Whether to use Ray for parallel processing (default: True)
        num_cpus: Number of CPUs to use for Ray (default: None, which means auto-detect)

    Returns:
        List of generated file paths
    """
    logger.warning("log_enhanced_plots_to_mlflow is deprecated. Use create_enhanced_visualizations instead.")
    
    # Call the new function which has all the functionality
    create_enhanced_visualizations(
        run_ids=run_ids,
        experiment_id=experiment_id,
        experiment_name="EnhancedPlots",
        run_number=run_number,
        use_ray=use_ray,
        num_cpus=num_cpus
    )
    
    # This is a bit of a hack, but we return an empty list since we can't easily
    # get the plot paths from create_enhanced_visualizations
    return []

@ray.remote
def process_plot_data(subgroup_key, subgroup_data, model_colors, show_markers, line_smoothing):
    """Process visualization data for one subgroup of models (Ray remote function)"""
    dataset_type = subgroup_data["dataset_type"]
    data_version = subgroup_data["data_version"]
    models = subgroup_data["models"]
    
    # Skip if no models have data
    if not models:
        return None
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot each model's train and validation loss
    for model_name, model_data in sorted(models.items()):
        color = model_colors[model_name]

        # Plot training loss
        if model_data["train"]:
            epochs = sorted(model_data["train"].keys())
            values = [model_data["train"][e] for e in epochs]

            # Apply smoothing if requested
            if line_smoothing > 0 and len(values) > 3:
                smooth_values = apply_smoothing(values, line_smoothing)
            else:
                smooth_values = values

            # For unsmoothed plots, we'll emphasize the markers
            if show_markers:
                # First plot the connecting line (thinner)
                plt.plot(
                    epochs,
                    values,  # Use original values always
                    linestyle="-",
                    color=color,
                    linewidth=1.0,
                    alpha=0.7,
                    label=f"{model_name}",
                )
                
                # Then add prominent markers
                plt.scatter(
                    epochs,
                    values,
                    marker="o",
                    s=40,  # Larger markers
                    color=color,
                    alpha=0.9,  # More visible
                    edgecolors="white",
                    linewidth=0.7,
                    zorder=10,  # Ensure markers are on top
                )
            else:
                # Just plot the line without markers
                line = plt.plot(
                    epochs,
                    values,  # Use original values, not smoothed
                    linestyle="-",
                    color=color,
                    linewidth=1.5,
                    label=f"{model_name}",
                    alpha=0.9,
                )[0]

    # Customize plot
    plt.title(f"Loss Curves - {dataset_type} ({data_version})", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Determine if log scale would be better
    y_values = []
    for model_data in models.values():
        y_values.extend(list(model_data["train"].values()))

    if y_values:
        y_array = np.array(y_values)
        if np.max(y_array) / np.min(y_array) > 10:
            plt.yscale("log")

    # Add legend - make it more compact if many models
    if len(models) > 5:
        plt.legend(fontsize=9, ncol=2, loc="best")
    else:
        plt.legend(fontsize=10, loc="best")

    plt.tight_layout()

    # Save plot to a temporary file
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plot_filename = f"temp_plot_{dataset_type}_{data_version}_{timestr}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        "subgroup_key": subgroup_key,
        "dataset_type": dataset_type,
        "data_version": data_version,
        "plot_filename": plot_filename
    }

def plot_loss_by_subgroups(
    run_ids: List[str],
    output_dir: str = None,
    show_markers: bool = True,
    line_smoothing: float = 0.0,
    use_ray: bool = True,
    num_cpus: Optional[int] = None
):
    """
    Create separate loss plots for each dataset/datatype subgroup,
    with each plot showing all models for that specific subgroup.

    Args:
        run_ids: List of MLflow run IDs to include
        output_dir: Directory to save output plots (defaults to time_series/plots if None)
        show_markers: Whether to show data points as markers
        line_smoothing: Smoothing factor (0.0 = raw data, higher values = more smoothing)
        use_ray: Whether to use Ray for parallel processing (default: True)
        num_cpus: Number of CPUs to use for Ray. If None, uses all available minus 2.

    Returns:
        List of paths to saved plots
    """
    logger.info("Generating loss plots by dataset/datatype subgroups")
    
    # Use time_series/plots if output_dir not specified
    if output_dir is None:
        time_series_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(time_series_dir, "plots")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get data from MLflow runs
    client = mlflow.tracking.MlflowClient()

    # Data structure to hold all run data
    all_data = {}

    # Extract data from runs
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)

            # Extract metadata
            model = run.data.params.get("model", "unknown")
            dataset_type = run.data.params.get("dataset_type", "unknown")
            data_version = run.data.params.get("data_version", "unknown")

            # Create subgroup key for this dataset/datatype combination
            subgroup_key = f"{dataset_type}_{data_version}"

            # Initialize subgroup entry if it doesn't exist
            if subgroup_key not in all_data:
                all_data[subgroup_key] = {
                    "dataset_type": dataset_type,
                    "data_version": data_version,
                    "models": {},
                }

            # Initialize model entry if it doesn't exist
            if model not in all_data[subgroup_key]["models"]:
                all_data[subgroup_key]["models"][model] = {"train": {}}

            # Extract epoch-specific metrics
            for metric_key, value in run.data.metrics.items():
                if metric_key.startswith("train_loss_epoch_"):
                    try:
                        epoch = int(metric_key.split("_")[-1])
                        all_data[subgroup_key]["models"][model]["train"][epoch] = value
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            logger.warning(f"Error processing run {run_id}: {e}")

    if not all_data:
        logger.warning("No loss data found in the provided runs")
        return []

    # Create a consistent color palette for models across all plots
    all_models = set()
    for subgroup_data in all_data.values():
        all_models.update(subgroup_data["models"].keys())

    model_colors = dict(
        zip(sorted(list(all_models)), sns.color_palette("husl", len(all_models)))
    )
    
    # Initialize Ray for parallel plot generation if requested and multiple subgroups exist
    plot_paths = []
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    if use_ray and len(all_data) > 1:
        try:
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                init_ray_for_mac(num_cpus)
                
            # Submit plotting tasks in parallel
            plot_tasks = [
                process_plot_data.remote(
                    subgroup_key, subgroup_data, model_colors, show_markers, line_smoothing
                )
                for subgroup_key, subgroup_data in sorted(all_data.items())
            ]
            
            # Get results
            plot_results = ray.get(plot_tasks)
            
            # Move temporary plots to final location
            for result in plot_results:
                if result is None:
                    continue
                
                temp_path = result["plot_filename"]
                final_path = f"{output_dir}/loss_plot_{result['dataset_type']}_{result['data_version']}_{timestr}.png"
                
                shutil.move(temp_path, final_path)
                plot_paths.append(final_path)
                logger.info(f"Created loss plot: {final_path}")
                
        except Exception as e:
            logger.warning(f"Failed to use Ray for parallel plot generation: {e}. Falling back to sequential processing.")
            use_ray = False  # Fall back to sequential processing
    
    # Sequential processing if Ray isn't available or there's only one subgroup
    if not use_ray:
        # Create one plot per subgroup
        for subgroup_key, subgroup_data in sorted(all_data.items()):
            dataset_type = subgroup_data["dataset_type"]
            data_version = subgroup_data["data_version"]
            models = subgroup_data["models"]

            # Skip if no models have data
            if not models:
                continue

            # Create figure
            plt.figure(figsize=(12, 6))

            # Plot each model's train and validation loss
            for model_name, model_data in sorted(models.items()):
                color = model_colors[model_name]

                # Plot training loss
                if model_data["train"]:
                    epochs = sorted(model_data["train"].keys())
                    values = [model_data["train"][e] for e in epochs]

                    # Apply smoothing if requested
                    if line_smoothing > 0 and len(values) > 3:
                        smooth_values = apply_smoothing(values, line_smoothing)
                    else:
                        smooth_values = values

                    # For unsmoothed plots, we'll emphasize the markers
                    if show_markers:
                        # First plot the connecting line (thinner)
                        plt.plot(
                            epochs,
                            values,  # Use original values always
                            linestyle="-",
                            color=color,
                            linewidth=1.0,
                            alpha=0.7,
                            label=f"{model_name}",
                        )
                        
                        # Then add prominent markers
                        plt.scatter(
                            epochs,
                            values,
                            marker="o",
                            s=40,  # Larger markers
                            color=color,
                            alpha=0.9,  # More visible
                            edgecolors="white",
                            linewidth=0.7,
                            zorder=10,  # Ensure markers are on top
                        )
                    else:
                        # Just plot the line without markers
                        line = plt.plot(
                            epochs,
                            values,  # Use original values, not smoothed
                            linestyle="-",
                            color=color,
                            linewidth=1.5,
                            label=f"{model_name}",
                            alpha=0.9,
                        )[0]

            # Customize plot
            plt.title(f"Loss Curves - {dataset_type} ({data_version})", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)

            # Determine if log scale would be better
            y_values = []
            for model_data in models.values():
                y_values.extend(list(model_data["train"].values()))

            if y_values:
                y_array = np.array(y_values)
                if np.max(y_array) / np.min(y_array) > 10:
                    plt.yscale("log")

            # Add legend - make it more compact if many models
            if len(models) > 5:
                plt.legend(fontsize=9, ncol=2, loc="best")
            else:
                plt.legend(fontsize=10, loc="best")

            plt.tight_layout()

            # Save plot
            plot_filename = (
                f"{output_dir}/loss_plot_{dataset_type}_{data_version}_{timestr}.png"
            )
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            plot_paths.append(plot_filename)
            logger.info(f"Created loss plot: {plot_filename}")

    return plot_paths

def create_enhanced_visualizations(run_ids, experiment_id, experiment_name, run_number=None, use_ray=True, num_cpus=None):
    """
    Create enhanced visualizations for the completed experiment runs.

    Args:
        run_ids: List of run IDs to visualize
        experiment_id: MLflow experiment ID
        experiment_name: Name of the experiment
        run_number: Run number (optional)
        use_ray: Whether to use Ray for parallel processing (default: True)
        num_cpus: Number of CPUs to use for Ray. If None, uses all available minus 2.
    """
    if not run_ids:
        logger.warning("No run IDs provided for enhanced visualizations")
        return

    logger.info("Generating enhanced loss plot visualizations...")
    
    # Get the time_series base directory
    time_series_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create output directory with experiment name and run number if provided
    safe_exp_name = experiment_name.replace(" ", "_").replace("/", "_")
    
    if run_number is not None:
        # Use organized structure with experiment name and run number
        output_dir = os.path.join(time_series_dir, "plots", safe_exp_name, f"run_{run_number}")
    else:
        # Fall back to timestamp-based directory for backward compatibility
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(time_series_dir, "plots", f"enhanced_{safe_exp_name}_{timestr}")
        
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Ray if requested
    if use_ray and not ray.is_initialized():
        try:
            init_ray_for_mac(num_cpus)
            logger.info("Ray initialized for enhanced visualization generation")
        except Exception as e:
            logger.warning(f"Failed to initialize Ray: {e}. Will use sequential processing.")
            use_ray = False

    # Generate loss plots using Ray-powered plotting
    plots = plot_loss_by_subgroups(
        run_ids=run_ids,
        output_dir=output_dir,
        show_markers=True,
        line_smoothing=0.0,
        use_ray=use_ray,
        num_cpus=num_cpus
    )

    # Log plots to MLflow
    if plots:
        try:
            # Make sure there's no active run that might conflict
            if mlflow.active_run():
                mlflow.end_run()
            
            # Generate a unique run name with timestamp
            timestr = time.strftime("%Y%m%d-%H%M%S")
            
            with mlflow.start_run(
                run_name=f"EnhancedPlots-{timestr}", experiment_id=experiment_id
            ):
                # Log artifacts
                for plot_path in plots:
                    mlflow.log_artifact(plot_path, "enhanced_plots")

                # Log metadata
                mlflow.set_tag("plot_type", "enhanced_loss_plots")
                mlflow.set_tag("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                mlflow.set_tag("generated_by", os.environ.get("USER", "keirparker"))
                mlflow.set_tag("runs_visualized", len(run_ids))
                mlflow.set_tag("is_summary", "true")
                
                # Add run number if provided
                if run_number is not None:
                    mlflow.set_tag("run_number", str(run_number))

                # Log the run IDs that were analyzed
                mlflow.log_param("analyzed_runs", ",".join(run_ids))

                logger.info(
                    f"Enhanced plots logged to MLflow with run ID: {mlflow.active_run().info.run_id}"
                )
        except Exception as e:
            logger.error(f"Error logging enhanced plots to MLflow: {e}")
            logger.info(f"Enhanced plots saved locally to {output_dir}")

        logger.info(f"Generated {len(plots)} enhanced visualization plots")
    else:
        logger.warning("No enhanced visualization plots were generated")


if __name__ == "__main__":
    print("Visualization utilities module loaded.")