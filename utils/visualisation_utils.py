#!/usr/bin/env python
"""
Visualization utilities for ML model training and evaluation.

This module provides high-quality visualization capabilities for machine learning
model training history, predictions, and integration with MLflow.

Author: GitHub Copilot for keirparker
Last updated: 2025-02-26
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

# Configure matplotlib for high-quality output
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)
mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Define color palette
COLOR_PALETTE = sns.color_palette("Paired", 15)


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


def _generate_plot_filename(model_name, dataset_type, data_version, plot_type):
    """Generate unique filename for a plot."""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    return f"plots/{model_name}_{dataset_type}_{data_version}_{plot_type}_{timestr}.png"


def plot_training_history(
    history: Dict,
    model_name: str,
    dataset_type: str,
    data_version: str,
    viz_config: Optional[VisualizationConfig] = None,
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
        model_name, dataset_type, data_version, "training_history"
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
            model_name, dataset_type, data_version, "epoch_times"
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
) -> str:
    """
    Generate visualization of model predictions with statistical analysis.

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

    # Plot the underlying true function if provided
    if true_func is not None:
        t_dense = np.linspace(
            min(min(t_train), min(t_test)), max(max(t_train), max(t_test)), 1000
        )
        y_dense = true_func(t_dense)
        ax1.plot(
            t_dense,
            y_dense,
            linestyle="--",
            color="red",
            alpha=0.7,
            linewidth=0.8,
            label="True Function",
        )

    # Plot training data as scatter points
    ax1.scatter(
        t_train,
        data_train,
        color=COLOR_PALETTE[1],
        s=20,
        alpha=0.5,
        edgecolor="k",
        linewidth=0.5,
        label="Training Data",
    )

    # Plot test data as scatter points with different style
    ax1.scatter(
        t_test,
        data_test,
        color=COLOR_PALETTE[2],
        s=10,
        alpha=0.6,
        edgecolor="k",
        linewidth=0.5,
        label="Test Data",
    )

    # Plot model predictions as a continuous line
    ax1.plot(
        t_test_sorted,
        predictions_sorted,
        color=COLOR_PALETTE[3],
        linewidth=1,
        label="Model Prediction",
    )

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

    # Calculate residuals
    residuals = data_test - predictions

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    mse = mean_squared_error(data_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(data_test, predictions)
    mae = mean_absolute_error(data_test, predictions)

    # Format metrics text
    metrics_text = f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}"

    # Create text box with metrics
    ax1.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Customize main plot
    ax1.set_title(f"Model Predictions - {model_name}")
    ax1.set_xlabel("Input")
    ax1.set_ylabel("Output")
    ax1.legend(loc="upper right")
    viz_config.style_axis(ax1)

    # Residual plot (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])

    # Plot residuals
    ax2.scatter(
        t_test,
        residuals,
        c=np.abs(residuals),
        cmap="YlOrRd",
        s=25,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
    )

    # Add zero reference line
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.7)

    ax2.set_title("Residuals")
    ax2.set_xlabel("Input")
    ax2.set_ylabel("Residual")
    viz_config.style_axis(ax2)

    # Residual histogram (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])

    # Create histogram with KDE
    sns.histplot(
        residuals, bins=15, kde=True, ax=ax3, color=COLOR_PALETTE[3], alpha=0.7
    )

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

    ax3.set_title("Residual Distribution")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Frequency")
    viz_config.style_axis(ax3)

    # Add overall title
    fig.suptitle(
        f"{model_name} on {dataset_type} ({data_version})", fontsize=14, y=0.98
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    plot_path = _generate_plot_filename(
        model_name, dataset_type, data_version, "predictions"
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
    output_dir: str = "plots",
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
        output_dir: Directory to save output plots
        show_markers: Whether to show data points as markers
        line_smoothing: Smoothing factor (0.0 = raw data, higher values = more smoothing)
        max_per_plot: Maximum number of curves to show in a single plot

    Returns:
        List of paths to saved plots
    """
    logger.info(f"Generating separated loss plots grouped by {group_by}")

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
    output_dir: str = "plots",
    show_markers: bool = True,
    line_smoothing: float = 0.0,
):
    """
    Create a grid of loss plots organized by dataset and data version.

    Args:
        run_ids: List of MLflow run IDs to include
        output_dir: Directory to save output plots
        show_markers: Whether to show data points as markers
        line_smoothing: Smoothing factor (0.0 = raw data, higher values = more smoothing)

    Returns:
        Path to the saved grid plot
    """
    logger.info("Generating loss plot grid")

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
    """
    Apply smoothing to a list of values.

    Args:
        values: List of values to smooth
        smoothing_factor: Smoothing factor (0-1, higher = more smoothing)

    Returns:
        Smoothed values
    """
    if smoothing_factor <= 0 or len(values) < 4:
        return values

    # Convert to numpy array for easier manipulation
    values_array = np.array(values)

    # Calculate window size based on smoothing factor
    # Higher smoothing factor means larger window
    smoothing_factor = min(0.9, max(0, smoothing_factor))  # Limit to 0-0.9
    window_size = max(3, int(len(values) * smoothing_factor))

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Apply convolution for smoothing
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(values_array, window, mode="same")

    # Fix edges (where convolution doesn't have full window)
    half_window = window_size // 2

    # Handle start of array
    for i in range(half_window):
        window_size_edge = i + half_window + 1
        if window_size_edge > 0:
            smoothed[i] = np.sum(values_array[:window_size_edge]) / window_size_edge

    # Handle end of array
    for i in range(half_window):
        idx = len(values) - i - 1
        window_size_edge = i + half_window + 1
        if window_size_edge > 0:
            smoothed[idx] = np.sum(values_array[-window_size_edge:]) / window_size_edge

    return smoothed


def log_enhanced_plots_to_mlflow(
    experiment_id: str,
    run_ids: List[str],
    output_dir: str = "plots",
    artifact_path: str = "enhanced_plots",
):
    """
    Generate and log enhanced loss plots to MLflow.

    Args:
        experiment_id: MLflow experiment ID
        run_ids: List of run IDs to include in visualization
        output_dir: Directory to save intermediate plots
        artifact_path: Path within MLflow artifacts to store plots

    Returns:
        List of generated file paths
    """
    logger.info("Generating and logging enhanced loss plots to MLflow")

    # Generate the different plot types
    plot_paths = []

    # Generate separated plots by dataset
    dataset_plots = plot_separate_loss_curves(
        run_ids,
        group_by="dataset_type",
        output_dir=output_dir,
        show_markers=False,
        line_smoothing=0.0,  # No smoothing
        max_per_plot=20,
    )
    plot_paths.extend(dataset_plots)

    # Generate separated plots by data version
    version_plots = plot_separate_loss_curves(
        run_ids,
        group_by="data_version",
        output_dir=output_dir,
        show_markers=False,
        line_smoothing=0.0,  # No smoothing
        max_per_plot=20,
    )
    plot_paths.extend(version_plots)

    # Generate a grid plot
    grid_plot = plot_loss_grid(
        run_ids,
        output_dir=output_dir,
        show_markers=True,
        line_smoothing=0.0,  # No smoothing
    )
    if grid_plot:
        plot_paths.append(grid_plot)

    # Log plots to MLflow as a separate "summary" run
    if plot_paths:
        timestr = time.strftime("%Y%m%d-%H%M%S")

        with mlflow.start_run(
            run_name=f"EnhancedPlots-{timestr}", experiment_id=experiment_id
        ):
            # Log artifacts
            for plot_path in plot_paths:
                mlflow.log_artifact(plot_path, artifact_path)

            # Log metadata
            mlflow.set_tag("plot_type", "enhanced_loss_plots")
            mlflow.set_tag("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            mlflow.set_tag("generated_by", os.environ.get("USER", "keirparker"))
            mlflow.set_tag("runs_visualized", len(run_ids))
            mlflow.set_tag("is_summary", "true")

            # Log the run IDs that were analyzed
            mlflow.log_param("analyzed_runs", ",".join(run_ids))

            logger.info(
                f"Enhanced plots logged to MLflow with run ID: {mlflow.active_run().info.run_id}"
            )

    return plot_paths


def plot_loss_by_subgroups(
    run_ids: List[str],
    output_dir: str = "plots",
    show_markers: bool = True,
    line_smoothing: float = 0.0,
):
    """
    Create separate loss plots for each dataset/datatype subgroup,
    with each plot showing all models for that specific subgroup.

    Args:
        run_ids: List of MLflow run IDs to include
        output_dir: Directory to save output plots
        show_markers: Whether to show data points as markers
        line_smoothing: Smoothing factor (0.0 = raw data, higher values = more smoothing)

    Returns:
        List of paths to saved plots
    """
    logger.info("Generating loss plots by dataset/datatype subgroups")

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

    # Generate plots - one for each subgroup
    plot_paths = []

    # Create a consistent color palette for models across all plots
    all_models = set()
    for subgroup_data in all_data.values():
        all_models.update(subgroup_data["models"].keys())

    model_colors = dict(
        zip(sorted(list(all_models)), sns.color_palette("husl", len(all_models)))
    )

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

                # Plot line
                line = plt.plot(
                    epochs,
                    smooth_values,
                    linestyle="-",
                    color=color,
                    linewidth=1,
                    label=f"{model_name}",
                    alpha=0.9,
                )[0]

                # Add markers if requested
                if show_markers:
                    plt.scatter(
                        epochs,
                        values,  # Original unsmoothed values
                        marker="o",
                        s=30,
                        color=color,
                        alpha=0.7,
                        edgecolors="white",
                        linewidth=0.5,
                    )

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
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = (
            f"{output_dir}/loss_plot_{dataset_type}_{data_version}_{timestr}.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

        plot_paths.append(plot_filename)
        logger.info(f"Created loss plot: {plot_filename}")

    return plot_paths


def apply_smoothing(values: List[float], smoothing_factor: float) -> np.ndarray:
    """
    Apply smoothing to a list of values.

    Args:
        values: List of values to smooth
        smoothing_factor: Smoothing factor (0-1, higher = more smoothing)

    Returns:
        Smoothed values
    """
    if smoothing_factor <= 0 or len(values) < 4:
        return values

    # Convert to numpy array for easier manipulation
    values_array = np.array(values)

    # Calculate window size based on smoothing factor
    # Higher smoothing factor means larger window
    smoothing_factor = min(0.9, max(0, smoothing_factor))  # Limit to 0-0.9
    window_size = max(3, int(len(values) * smoothing_factor))

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Apply convolution for smoothing
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(values_array, window, mode="same")

    # Fix edges (where convolution doesn't have full window)
    half_window = window_size // 2

    # Handle start of array
    for i in range(half_window):
        window_size_edge = i + half_window + 1
        if window_size_edge > 0:
            smoothed[i] = np.sum(values_array[:window_size_edge]) / window_size_edge

    # Handle end of array
    for i in range(half_window):
        idx = len(values) - i - 1
        window_size_edge = i + half_window + 1
        if window_size_edge > 0:
            smoothed[idx] = np.sum(values_array[-window_size_edge:]) / window_size_edge

    return smoothed


def log_enhanced_plots_to_mlflow(
    experiment_id: str,
    run_ids: List[str],
    output_dir: str = "plots",
    artifact_path: str = "enhanced_plots",
):
    """
    Generate and log enhanced loss plots to MLflow.

    Args:
        experiment_id: MLflow experiment ID
        run_ids: List of run IDs to include in visualization
        output_dir: Directory to save intermediate plots
        artifact_path: Path within MLflow artifacts to store plots

    Returns:
        List of generated file paths
    """
    logger.info("Generating and logging enhanced loss plots to MLflow")

    # Generate loss plots by dataset/datatype subgroups
    plot_paths = plot_loss_by_subgroups(
        run_ids,
        output_dir=output_dir,
        show_markers=False,
        line_smoothing=0.0,  # No smoothing for clearer data visualization
    )

    # Log plots to MLflow as a separate "summary" run
    if plot_paths:
        timestr = time.strftime("%Y%m%d-%H%M%S")

        with mlflow.start_run(
            run_name=f"EnhancedPlots-{timestr}", experiment_id=experiment_id
        ):
            # Log artifacts
            for plot_path in plot_paths:
                mlflow.log_artifact(plot_path, artifact_path)

            # Log metadata
            mlflow.set_tag("plot_type", "enhanced_loss_plots")
            mlflow.set_tag("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            mlflow.set_tag("generated_by", os.environ.get("USER", "keirparker"))
            mlflow.set_tag("runs_visualized", len(run_ids))
            mlflow.set_tag("is_summary", "true")

            # Log the run IDs that were analyzed
            mlflow.log_param("analyzed_runs", ",".join(run_ids))

            logger.info(
                f"Enhanced plots logged to MLflow with run ID: {mlflow.active_run().info.run_id}"
            )

    return plot_paths

def create_enhanced_visualizations(run_ids, experiment_id, experiment_name):
    """
    Create enhanced visualizations for the completed experiment runs.

    Args:
        run_ids: List of run IDs to visualize
        experiment_id: MLflow experiment ID
        experiment_name: Name of the experiment
    """
    if not run_ids:
        logger.warning("No run IDs provided for enhanced visualizations")
        return

    logger.info("Generating enhanced loss plot visualizations...")

    # Create output directory with timestamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"plots/enhanced_{experiment_name.replace(' ', '_')}_{timestr}"
    os.makedirs(output_dir, exist_ok=True)

    # Generate and log enhanced plots
    plots = log_enhanced_plots_to_mlflow(
        experiment_id=experiment_id,
        run_ids=run_ids,
        output_dir=output_dir,
        artifact_path="enhanced_plots",
    )

    if plots:
        logger.info(f"Generated {len(plots)} enhanced visualization plots")
    else:
        logger.warning("No enhanced visualization plots were generated")




if __name__ == "__main__":
    print("Visualization utilities module loaded.")
