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

# Configure matplotlib for high-quality output
plt.style.use("seaborn-v0_8-whitegrid")
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

    epochs = range(1, len(history["train_loss"]) + 1)
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
        ax1.plot(
            epochs,
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
        ax2.plot(
            epochs,
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
        ax3.plot(
            epochs,
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
            ax4.plot(
                epochs,
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
            ax4.plot(
                epochs,
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
        plt.plot(
            epochs,
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
            valid_epochs = list(epochs)[window_size - 1 :]
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
            fig.add_trace(
                go.Scatter(
                    x=list(epochs),
                    y=history["train_loss"],
                    mode="lines+markers",
                    name="Training Loss",
                    line=dict(color="rgb(31, 119, 180)"),
                ),
                row=1,
                col=1,
            )

            if "val_loss" in history and history["val_loss"] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=list(epochs),
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


if __name__ == "__main__":
    print("Visualization utilities module loaded.")
