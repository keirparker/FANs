#!/usr/bin/env python
"""
Model comparison and analysis script for ML experiments.

This script analyzes MLflow runs to generate comparative visualizations,
summary tables, and rankings across different models and datasets.

Author: GitHub Copilot for keirparker
Created: 2025-02-26 16:32:55
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient


def setup_logger():
    """Configure the logger."""
    logger.remove()
    logger.add(
        "logs/analysis_{time}.log",
        rotation="500 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(
        lambda msg: print(msg),
        level="INFO",
        format="{level} | <blue>{time:HH:mm:ss}</blue> | <level>{message}</level>"
    )


def compare_models(run_ids=None, experiment_id=None, metric="test_r2", output_dir="results"):
    """
    Generate comparative visualizations across different models.

    Args:
        run_ids: List of MLflow run IDs to compare (if None, uses latest experiment)
        experiment_id: MLflow experiment ID to use (if run_ids is None)
        metric: Metric to compare models on
        output_dir: Directory to save output files

    Returns:
        str: Path to the saved comparison plot
    """
    try:
        logger.info(f"Generating model comparison for metric: {metric}")
        client = MlflowClient()

        # Get run data
        runs_data = []

        if run_ids:
            for run_id in run_ids:
                try:
                    run = client.get_run(run_id)
                    runs_data.append(run)
                except:
                    logger.warning(f"Could not find run with ID: {run_id}")
        elif experiment_id:
            runs = client.search_runs(experiment_ids=[experiment_id])
            runs_data = runs
        else:
            # Get the current experiment
            current_experiment = mlflow.get_experiment_by_name("Default")
            if current_experiment:
                runs = client.search_runs(experiment_ids=[current_experiment.experiment_id])
                runs_data = runs
            else:
                logger.warning("No default experiment found")

        if not runs_data:
            logger.warning("No runs found for comparison")
            return None

        # Extract data for plotting
        data = []
        for run in runs_data:
            if metric in run.data.metrics:
                model_name = run.data.params.get("model", "unknown")
                dataset = run.data.params.get("dataset_type", "unknown")
                data_version = run.data.params.get("data_version", "unknown")
                metric_value = run.data.metrics[metric]

                data.append({
                    "model": model_name,
                    "dataset": dataset,
                    "data_version": data_version,
                    "metric": metric_value,
                    "run_id": run.info.run_id
                })

        if not data:
            logger.warning(f"No data found for metric: {metric}")
            return None

        # Create dataframe for visualization
        df = pd.DataFrame(data)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create visualization
        plt.figure(figsize=(12, 8))

        # For R2, higher is better, for others lower is better
        is_inverse = metric != "test_r2"

        # Find best model for highlighting
        if is_inverse:
            best_idx = df.groupby('dataset')['metric'].idxmin()
        else:
            best_idx = df.groupby('dataset')['metric'].idxmax()

        best_models = df.loc[best_idx]

        # Create grouped bar chart
        g = sns.catplot(
            data=df,
            x="model",
            y="metric",
            hue="data_version",
            col="dataset",
            kind="bar",
            height=6,
            aspect=0.8,
            legend=True,
            palette="viridis"
        )

        # Set titles and labels
        metric_display = metric.replace("test_", "").upper()
        if metric_display == "R2":
            metric_display = "R²"

        g.set_axis_labels("Model", f"{metric_display}")
        g.set_titles("Dataset: {col_name}")

        # Add value labels and highlight best models
        for ax_i, dataset in zip(g.axes.flat, sorted(df["dataset"].unique())):
            # Add value labels
            for p in ax_i.patches:
                ax_i.annotate(
                    f"{p.get_height():.4f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45
                )

            # Highlight best model for this dataset
            best_for_dataset = best_models[best_models['dataset'] == dataset]
            if not best_for_dataset.empty:
                best_model = best_for_dataset['model'].values[0]
                best_version = best_for_dataset['data_version'].values[0]
                best_value = best_for_dataset['metric'].values[0]

                # Find the patch to highlight
                for p in ax_i.patches:
                    # Check if this patch corresponds to the best model
                    patch_x = p.get_x() + p.get_width() / 2
                    model_idx = ax_i.get_xticks()[int(patch_x)]
                    if (ax_i.get_xticklabels()[model_idx].get_text() == best_model and
                            abs(p.get_height() - best_value) < 1e-6):
                        # Highlight this patch
                        p.set_edgecolor('gold')
                        p.set_linewidth(2)
                        # Add a star
                        ax_i.annotate(
                            "★",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            color='gold'
                        )

        plt.suptitle(f"Model Comparison - {metric_display}", fontsize=16)
        plt.tight_layout()

        # Generate file name
        timestr = time.strftime("%Y%m%d-%H%M%S")
        comparison_path = f"{output_dir}/model_comparison_{metric}_{timestr}.png"
        plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
        plt.close()

        logger.info(f"Model comparison plot saved to {comparison_path}")
        return comparison_path

    except Exception as e:
        logger.error(f"Error generating model comparison: {e}")
        return None

def generate_model_summary_table(run_ids, experiment_name):
    """
    Generate a summary table of model performance and log it to MLflow
    under the same experiment as the runs.

    Args:
        run_ids: List of MLflow run IDs
        experiment_name: Name of the experiment

    Returns:
        pd.DataFrame: The summary table
    """
    try:
        import pandas as pd
        from mlflow.tracking import MlflowClient

        logger.info("Generating model performance summary table")
        client = MlflowClient()

        # Get the experiment ID from the first run (all runs should be in the same experiment)
        if run_ids:
            experiment_id = None
            try:
                first_run = client.get_run(run_ids[0])
                experiment_id = first_run.info.experiment_id
                logger.info(f"Using experiment ID: {experiment_id} from run {run_ids[0]}")
            except Exception as e:
                logger.warning(f"Could not get experiment ID from run: {e}")

            # If we couldn't get the experiment ID from the run, try to get it by name
            if experiment_id is None:
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment:
                        experiment_id = experiment.experiment_id
                        logger.info(f"Using experiment ID: {experiment_id} from experiment name")
                except Exception as e:
                    logger.warning(f"Could not get experiment ID by name: {e}")
        else:
            logger.warning("No run IDs provided")
            return None

        metrics_to_include = ["test_r2", "test_rmse", "test_mae", "test_mape", "training_time_seconds"]

        # Get runs data
        runs_data = []
        for run_id in run_ids:
            try:
                run = client.get_run(run_id)
                runs_data.append(run)
            except Exception as e:
                logger.warning(f"Could not find run with ID: {run_id}, error: {e}")

        if not runs_data:
            logger.warning("No runs found for summary table")
            return None

        # Extract data for table
        table_data = []
        for run in runs_data:
            run_data = {
                "run_id": run.info.run_id,
                "model": run.data.params.get("model", "unknown"),
                "dataset": run.data.params.get("dataset_type", "unknown"),
                "data_version": run.data.params.get("data_version", "unknown"),
                "status": run.data.tags.get("status", run.info.status),
            }

            # Add metrics
            for metric in metrics_to_include:
                if metric in run.data.metrics:
                    run_data[metric] = run.data.metrics[metric]
                else:
                    run_data[metric] = None

            # Add epochs information
            if "epochs" in run.data.params:
                run_data["epochs"] = run.data.params["epochs"]

            # Add final loss values
            for loss_type in ["train_loss", "val_loss"]:
                if f"final_{loss_type}" in run.data.metrics:
                    run_data[f"final_{loss_type}"] = run.data.metrics[f"final_{loss_type}"]

            table_data.append(run_data)

        if not table_data:
            logger.warning("No data available for summary table")
            return None

        # Create DataFrame
        df = pd.DataFrame(table_data)

        # Sort by dataset, model and data version
        if all(col in df.columns for col in ["dataset", "model", "data_version"]):
            df = df.sort_values(by=["dataset", "model", "data_version"])

        # Add timestamp and user information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        username = os.environ.get('USER', 'keirparker')

        # Generate file paths with timestamp to avoid overwriting
        timestr = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("results", exist_ok=True)

        html_path = f"results/summary_table_{experiment_name.replace(' ', '_')}_{timestr}.html"
        csv_path = f"results/summary_table_{experiment_name.replace(' ', '_')}_{timestr}.csv"

        # Create styled HTML
        styled_df = df.style.set_caption(f"Model Performance Summary - {experiment_name}")

        # Highlight best values for each dataset
        for dataset in df["dataset"].unique():
            dataset_mask = df["dataset"] == dataset
            for metric in metrics_to_include:
                if metric in df.columns and metric != "training_time_seconds":
                    if metric == "test_r2":  # Higher is better
                        best_idx = df.loc[dataset_mask, metric].idxmax()
                        if pd.notna(best_idx):
                            styled_df = styled_df.highlight_max(
                                subset=pd.IndexSlice[pd.Index([best_idx]), [metric]],
                                color="lightgreen"
                            )
                    else:  # Lower is better
                        best_idx = df.loc[dataset_mask, metric].idxmin()
                        if pd.notna(best_idx):
                            styled_df = styled_df.highlight_min(
                                subset=pd.IndexSlice[pd.Index([best_idx]), [metric]],
                                color="lightgreen"
                            )

        # Format numeric columns
        numeric_format = {
            "test_r2": "{:.4f}",
            "test_rmse": "{:.4f}",
            "test_mae": "{:.4f}",
            "test_mape": "{:.2f}",
            "training_time_seconds": "{:.2f}",
            "final_train_loss": "{:.6f}",
            "final_val_loss": "{:.6f}"
        }

        for col, fmt in numeric_format.items():
            if col in df.columns:
                styled_df = styled_df.format({col: fmt})

        # Save files
        styled_df.to_html(html_path)
        df.to_csv(csv_path)

        # Now log to MLflow under the same experiment
        if experiment_id:
            # Log as a special "summary" run within the same experiment
            with mlflow.start_run(run_name=f"Summary-{timestr}", experiment_id=experiment_id):
                # Log artifacts
                mlflow.log_artifact(html_path)
                mlflow.log_artifact(csv_path)

                # Log metadata
                mlflow.set_tag("summary_type", "model_comparison")
                mlflow.set_tag("table_generated_at", timestamp)
                mlflow.set_tag("table_generated_by", username)
                mlflow.set_tag("runs_analyzed", len(df))
                mlflow.set_tag("is_summary", "true")  # Easy tag to filter for summaries

                # Log the run IDs that were analyzed
                mlflow.log_param("analyzed_runs", ",".join(run_ids))

                logger.info(f"Summary table logged to MLflow experiment '{experiment_name}' and saved to {html_path}")
                logger.info(f"Summary run ID: {mlflow.active_run().info.run_id}")
        else:
            logger.warning("Could not log to MLflow - no experiment ID found")
            logger.info(f"Summary table saved locally to {html_path}")

        return df

    except Exception as e:
        logger.error(f"Error generating summary table: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    

def generate_model_rankings(run_ids=None, experiment_id=None, metrics=None, output_dir="results"):
    """
    Generate ranking tables and visualizations for models across datasets.

    Args:
        run_ids: List of MLflow run IDs
        experiment_id: MLflow experiment ID to use (if run_ids is None)
        metrics: List of metrics to rank by
        output_dir: Directory to save output files

    Returns:
        dict: Paths to generated ranking files
    """
    if metrics is None:
        metrics = ["test_r2", "test_rmse", "test_mae"]

    try:
        logger.info("Generating model rankings across datasets and metrics")
        client = MlflowClient()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get runs data first
        runs_data = []

        if run_ids:
            for run_id in run_ids:
                try:
                    run = client.get_run(run_id)
                    runs_data.append(run)
                except:
                    logger.warning(f"Could not find run with ID: {run_id}")
        elif experiment_id:
            runs = client.search_runs(experiment_ids=[experiment_id])
            runs_data = runs
        else:
            # Get the current experiment
            current_experiment = mlflow.get_experiment_by_name("Default")
            if current_experiment:
                runs = client.search_runs(experiment_ids=[current_experiment.experiment_id])
                runs_data = runs
            else:
                logger.warning("No default experiment found")

        if not runs_data:
            logger.warning("No runs found for rankings")
            return {}

        # Extract data for all runs
        data = []
        for run in runs_data:
            run_data = {
                "Run ID": run.info.run_id,
                "Model": run.data.params.get("model", "unknown"),
                "Dataset": run.data.params.get("dataset_type", "unknown"),
                "Data Version": run.data.params.get("data_version", "unknown")
            }

            # Add metrics
            has_metrics = False
            for metric in metrics:
                if metric in run.data.metrics:
                    run_data[metric] = run.data.metrics[metric]
                    has_metrics = True

            if has_metrics:
                data.append(run_data)

        if not data:
            logger.warning("No metric data available for rankings")
            return {}

        # Create DataFrame with all data
        df = pd.DataFrame(data)

        # Generate timestamp for filenames
        timestr = time.strftime("%Y%m%d-%H%M%S")

        # Create rankings for each dataset and metric
        rankings = []
        model_scores = {}  # For overall score calculation

        # Process each dataset type
        for dataset in df["Dataset"].unique():
            dataset_df = df[df["Dataset"] == dataset].copy()

            # Process each metric
            for metric in metrics:
                if metric not in dataset_df.columns:
                    continue

                # Create ranking for this metric (by model and data version)
                dataset_metric_df = dataset_df[["Model", "Data Version", metric]].dropna().copy()

                # Skip if no data
                if len(dataset_metric_df) == 0:
                    continue

                # Determine ranking order (ascending for error metrics, descending for score metrics)
                ascending = metric != "test_r2"  # True for RMSE/MAE, False for R²

                # Rank models (lower rank = better)
                dataset_metric_df["Rank"] = dataset_metric_df[metric].rank(
                    ascending=ascending, method="min")

                # Add dataset and metric info
                dataset_metric_df["Dataset"] = dataset
                dataset_metric_df["Metric"] = metric

                # Calculate normalized score (0-100) where higher is always better
                if ascending:  # For error metrics (lower is better)
                    max_val = dataset_metric_df[metric].max()
                    min_val = dataset_metric_df[metric].min()
                    range_val = max_val - min_val if max_val > min_val else 1
                    dataset_metric_df["Score"] = 100 * (1 - (dataset_metric_df[metric] - min_val) / range_val)
                else:  # For R² (higher is better)
                    max_val = dataset_metric_df[metric].max()
                    min_val = dataset_metric_df[metric].min()
                    range_val = max_val - min_val if max_val > min_val else 1
                    dataset_metric_df["Score"] = 100 * (dataset_metric_df[metric] - min_val) / range_val

                # Add to model overall scores
                for _, row in dataset_metric_df.iterrows():
                    key = (row["Model"], row["Data Version"])
                    if key not in model_scores:
                        model_scores[key] = []
                    model_scores[key].append(row["Score"])

                rankings.append(dataset_metric_df)

        if not rankings:
            logger.warning("No ranking data generated")
            return {}

        # Combine all rankings
        rankings_df = pd.concat(rankings)

        # Calculate overall scores
        overall_scores = []
        for (model, data_version), scores in model_scores.items():
            overall_scores.append({
                "Model": model,
                "Data Version": data_version,
                "Overall Score": np.mean(scores),
                "Rank": 0  # Will be set later
            })

        # Create overall score dataframe and set ranks
        overall_df = pd.DataFrame(overall_scores)
        overall_df["Rank"] = overall_df["Overall Score"].rank(ascending=False, method="min")

        # Save rankings and overall scores
        all_paths = {}

        # Save detailed rankings
        rankings_path = f"{output_dir}/model_rankings_{timestr}.csv"
        rankings_df.to_csv(rankings_path, index=False)
        all_paths["rankings_csv"] = rankings_path
        logger.info(f"Model rankings saved as CSV: {rankings_path}")

        # Save overall scores
        overall_path = f"{output_dir}/model_overall_scores_{timestr}.csv"
        overall_df.to_csv(overall_path, index=False)
        all_paths["overall_scores_csv"] = overall_path
        logger.info(f"Model overall scores saved as CSV: {overall_path}")

        # Format the ranking table - pivot for better visualization
        table_df = rankings_df.pivot_table(
            index=["Model", "Data Version"],
            columns=["Dataset", "Metric"],
            values=["Rank", "Score"],
            aggfunc="first"
        )

        # Save pivot table
        rankings_pivot_path = f"{output_dir}/model_rankings_pivot_{timestr}.csv"
        table_df.to_csv(rankings_pivot_path)
        all_paths["rankings_pivot_csv"] = rankings_pivot_path
        logger.info(f"Pivot table of rankings saved: {rankings_pivot_path}")

        # Generate ranking visualization - heatmap of ranks
        plt.figure(figsize=(14, 10))

        # Extract rank values from the pivot table
        try:
            rank_df = table_df.xs('Rank', axis=1, level=0)

            # Create the heatmap
            ax = sns.heatmap(rank_df, annot=True, cmap="YlGnBu_r", fmt=".0f",
                             linewidths=.5, cbar_kws={"label": "Rank (lower is better)"})

            plt.title("Model Rankings by Dataset and Metric", fontsize=16)
            plt.tight_layout()

            # Save ranking heatmap
            ranking_viz_path = f"{output_dir}/ranking_heatmap_{timestr}.png"
            plt.savefig(ranking_viz_path, bbox_inches='tight', dpi=300)
            plt.close()

            all_paths["ranking_heatmap"] = ranking_viz_path
            logger.info(f"Ranking heatmap saved: {ranking_viz_path}")
        except Exception as e:
            logger.error(f"Error creating ranking heatmap: {e}")

        # Generate overall score visualization - horizontal bar chart
        plt.figure(figsize=(12, 8))

        # Sort by overall score
        plot_df = overall_df.sort_values("Overall Score", ascending=False)

        # Create bar chart with model names and data versions
        plot_df["Label"] = plot_df["Model"] + " (" + plot_df["Data Version"] + ")"

        # Use different colors based on data version
        data_versions = plot_df["Data Version"].unique()
        colors = sns.color_palette("viridis", n_colors=len(data_versions))
        color_map = {version: color for version, color in zip(data_versions, colors)}
        bar_colors = [color_map[version] for version in plot_df["Data Version"]]

        # Create horizontal bar chart
        ax = sns.barplot(
            x="Overall Score",
            y="Label",
            data=plot_df,
            palette=bar_colors,
            orient="h"
        )

        # Add rank labels
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                row["Overall Score"] + 1,
                i,
                f"Rank: {int(row['Rank'])}",
                va='center'
            )

        plt.title("Overall Model Performance (Higher is Better)", fontsize=16)
        plt.xlabel("Overall Score (0-100)")
        plt.ylabel("Model")
        plt.tight_layout()

        # Add legend for data versions
        legend_handles = [plt.Rectangle((0,0), 1, 1, color=color_map[v]) for v in data_versions]
        plt.legend(legend_handles, data_versions, title="Data Version", loc="lower right")

        # Save overall score visualization
        overall_viz_path = f"{output_dir}/overall_scores_{timestr}.png"
        plt.savefig(overall_viz_path, bbox_inches='tight', dpi=300)
        plt.close()

        all_paths["overall_scores_viz"] = overall_viz_path
        logger.info(f"Overall scores visualization saved: {overall_viz_path}")

        # Generate an HTML report with all visualizations
        html_report_path = f"{output_dir}/model_rankings_report_{timestr}.html"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Rankings and Performance Analysis</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 30px;
                    line-height: 1.6;
                }}
                h1, h2 {{
                    color: #2c3e50;
                }}
                h1 {{
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .section {{
                    margin-top: 30px;
                    margin-bottom: 40px;
                }}
                .viz-container {{
                    text-align: center;
                    margin-top: 20px;
                }}
                .viz-container img {{
                    max-width: 100%;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-radius: 5px;
                }}
                .caption {{
                    font-style: italic;
                    color: #7f8c8d;
                    margin-top: 10px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .footer {{
                    margin-top: 40px;
                    font-size: 0.8em;
                    color: #7f8c8d;
                    text-align: right;
                    border-top: 1px solid #eee;
                    padding-top: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>Model Rankings and Performance Analysis</h1>

            <div class="section">
                <h2>Overall Model Performance</h2>
                <p>This chart shows the overall performance of each model across all datasets and metrics.
                   Models are ranked by their average score (0-100) where higher is better.</p>
                <div class="viz-container">
                    <img src="{os.path.basename(overall_viz_path)}" alt="Overall Model Scores">
                    <div class="caption">Higher score indicates better overall performance across all metrics and datasets.</div>
                </div>
            </div>

            <div class="section">
                <h2>Model Rankings by Dataset and Metric</h2>
                <p>This heatmap shows how each model ranks for each combination of dataset and metric.
                   Lower rank number (darker color) indicates better performance.</p>
                <div class="viz-container">
                    <img src="{os.path.basename(ranking_viz_path)}" alt="Model Rankings Heatmap">
                    <div class="caption">Rank values: 1 is best, higher numbers indicate worse performance.</div>
                </div>
            </div>

            <div class="section">
                <h2>Top 5 Models Overall</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Data Version</th>
                        <th>Overall Score</th>
                    </tr>
        """

        # Add top 5 models to table
        top_models = overall_df.sort_values("Overall Score", ascending=False).head(5)
        for _, row in top_models.iterrows():
            html += f"""
                    <tr>
                        <td>{int(row['Rank'])}</td>
                        <td>{row['Model']}</td>
                        <td>{row['Data Version']}</td>
                        <td>{row['Overall Score']:.2f}</td>
                    </tr>
            """

        html += f"""
                </table>
            </div>

            <div class="footer">
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: {os.environ.get('USER', 'keirparker')}</p>
            </div>
        </body>
        </html>
        """

        with open(html_report_path, 'w') as f:
            f.write(html)

        all_paths["html_report"] = html_report_path
        logger.info(f"Comprehensive rankings report saved: {html_report_path}")

        return all_paths

    except Exception as e:
        logger.error(f"Error generating model rankings: {e}")
        return {}


def plot_learning_curves(
    run_ids=None, experiment_id=None, metric="train_loss", output_dir="results"
):
    """
    Generate a plot comparing learning curves across different models.

    Args:
        run_ids: List of MLflow run IDs
        experiment_id: MLflow experiment ID to use (if run_ids is None)
        metric: Metric to plot (train_loss, val_loss, etc.)
        output_dir: Directory to save output files

    Returns:
        str: Path to the saved learning curves plot
    """
    try:
        logger.info(f"Generating learning curve comparison for metric: {metric}")
        client = MlflowClient()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get runs data
        runs_data = []

        if run_ids:
            for run_id in run_ids:
                try:
                    run = client.get_run(run_id)
                    runs_data.append(run)
                except:
                    logger.warning(f"Could not find run with ID: {run_id}")
        elif experiment_id:
            runs = client.search_runs(experiment_ids=[experiment_id])
            runs_data = runs
        else:
            # Get the current experiment
            current_experiment = mlflow.get_experiment_by_name("Default")
            if current_experiment:
                runs = client.search_runs(
                    experiment_ids=[current_experiment.experiment_id]
                )
                runs_data = runs
            else:
                logger.warning("No default experiment found")

        if not runs_data:
            logger.warning("No runs found for learning curve comparison")
            return None

        # Extract learning curve data for each run
        curve_data = {}
        max_epochs = 0

        for run in runs_data:
            run_id = run.info.run_id
            model_name = run.data.params.get("model", "unknown")
            dataset = run.data.params.get("dataset_type", "unknown")
            data_version = run.data.params.get("data_version", "unknown")

            # Create a label for the run
            run_label = f"{model_name} - {dataset} ({data_version})"

            # Extract epoch data
            epoch_values = {}
            for metric_key in run.data.metrics:
                if metric_key.startswith(f"{metric}_epoch_"):
                    try:
                        epoch_num = int(metric_key.split("_")[-1])
                        epoch_values[epoch_num] = run.data.metrics[metric_key]
                        max_epochs = max(max_epochs, epoch_num)
                    except ValueError:
                        continue

            if epoch_values:
                curve_data[run_label] = epoch_values

        if not curve_data:
            logger.warning(f"No epoch data found for metric: {metric}")
            return None

        # Create the learning curve plot
        plt.figure(figsize=(14, 8))

        # Use a good color palette
        colors = sns.color_palette("husl", len(curve_data))

        # Plot each learning curve
        for i, (run_label, epochs_data) in enumerate(curve_data.items()):
            epochs = list(epochs_data.keys())
            values = list(epochs_data.values())

            # Sort by epoch number
            sorted_data = sorted(zip(epochs, values))
            epochs = [e for e, _ in sorted_data]
            values = [v for _, v in sorted_data]

            plt.plot(
                epochs,
                values,
                marker="o",
                markersize=4,
                linewidth=2,
                label=run_label,
                color=colors[i],
            )

        # Make the plot nicer
        plt.xlabel("Epoch", fontsize=12)

        # Format y-label based on metric
        metric_label = metric.replace("_", " ").title()
        plt.ylabel(metric_label, fontsize=12)

        # Set title
        plt.title(f"Learning Curves - {metric_label}", fontsize=14)

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add legend with smaller font size if many runs
        if len(curve_data) > 10:
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.legend(fontsize=10)

        # Adjust layout for legend
        plt.tight_layout()

        # Generate filename
        timestr = time.strftime("%Y%m%d-%H%M%S")
        learning_curve_path = f"{output_dir}/learning_curves_{metric}_{timestr}.png"

        # Save plot
        plt.savefig(learning_curve_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Learning curves plot saved to {learning_curve_path}")
        return learning_curve_path

    except Exception as e:
        logger.error(f"Error generating learning curves plot: {e}")
        return None


def plot_prediction_comparison(
    run_ids=None, experiment_id=None, dataset_type=None, output_dir="results"
):
    """
    Generate a plot comparing predictions across different models for the same dataset.

    Args:
        run_ids: List of MLflow run IDs
        experiment_id: MLflow experiment ID to use (if run_ids is None)
        dataset_type: Filter runs by specific dataset type
        output_dir: Directory to save output files

    Returns:
        str: Path to the saved prediction comparison plot
    """
    try:
        logger.info("Generating model prediction comparison plot")

        # This would be a placeholder since getting actual prediction data
        # would require the models to be loaded and re-run on test data

        # In a real implementation, this would either:
        # 1. Load saved prediction data from MLflow artifacts
        # 2. Load models and generate new predictions on test data
        # 3. Use prediction plots saved as artifacts

        logger.warning(
            "Prediction comparison requires saved prediction data, currently not implemented"
        )
        return None

    except Exception as e:
        logger.error(f"Error generating prediction comparison: {e}")
        return None


def get_run_ids_from_experiment(
    experiment_name=None, experiment_id=None, filter_kwargs=None
):
    """
    Helper function to get run IDs from an experiment.

    Args:
        experiment_name: Name of the experiment
        experiment_id: ID of the experiment
        filter_kwargs: Additional filtering criteria

    Returns:
        list: List of run IDs
    """
    try:
        client = MlflowClient()

        # Get experiment ID
        if experiment_id is None and experiment_name is not None:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
            else:
                logger.warning(f"Experiment '{experiment_name}' not found")
                return []

        if experiment_id is None:
            logger.warning("No experiment specified")
            return []

        # Set up filter string
        filter_string = ""
        if filter_kwargs:
            conditions = []
            for key, value in filter_kwargs.items():
                conditions.append(f'tags."{key}" = "{value}"')
            if conditions:
                filter_string = " and ".join(conditions)

        # Get runs
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string if filter_string else None,
        )

        # Extract run IDs
        run_ids = [run.info.run_id for run in runs]
        logger.info(f"Found {len(run_ids)} runs in experiment")
        return run_ids

    except Exception as e:
        logger.error(f"Error getting run IDs from experiment: {e}")
        return []
