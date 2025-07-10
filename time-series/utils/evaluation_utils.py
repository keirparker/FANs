#!/usr/bin/env python
"""
Model comparison and analysis script for ML experiments.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from loguru import logger
import logging
import mlflow
from mlflow.tracking import MlflowClient
import ray


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
            current_experiment = mlflow.get_experiment_by_name("Default")
            if current_experiment:
                runs = client.search_runs(experiment_ids=[current_experiment.experiment_id])
                runs_data = runs
            else:
                logger.warning("No default experiment found")

        if not runs_data:
            logger.warning("No runs found for comparison")
            return None

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

        df = pd.DataFrame(data)

        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))

        is_inverse = metric != "test_r2"

        if is_inverse:
            best_idx = df.groupby('dataset')['metric'].idxmin()
        else:
            best_idx = df.groupby('dataset')['metric'].idxmax()

        best_models = df.loc[best_idx]

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

        metric_display = metric.replace("test_", "").upper()
        if metric_display == "R2":
            metric_display = "R²"

        g.set_axis_labels("Model", f"{metric_display}")
        g.set_titles("Dataset: {col_name}")

        for ax_i, dataset in zip(g.axes.flat, sorted(df["dataset"].unique())):
            for p in ax_i.patches:
                ax_i.annotate(
                    f"{p.get_height():.4f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45
                )

            best_for_dataset = best_models[best_models['dataset'] == dataset]
            if not best_for_dataset.empty:
                best_model = best_for_dataset['model'].values[0]
                best_version = best_for_dataset['data_version'].values[0]
                best_value = best_for_dataset['metric'].values[0]

                for p in ax_i.patches:
                    patch_x = p.get_x() + p.get_width() / 2
                    model_idx = ax_i.get_xticks()[int(patch_x)]
                    if (ax_i.get_xticklabels()[model_idx].get_text() == best_model and
                            abs(p.get_height() - best_value) < 1e-6):
                        p.set_edgecolor('gold')
                        p.set_linewidth(2)
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

        timestr = time.strftime("%Y%m%d-%H%M%S")
        comparison_path = f"{output_dir}/model_comparison_{metric}_{timestr}.png"
        plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
        plt.close()

        logger.info(f"Model comparison plot saved to {comparison_path}")
        return comparison_path

    except Exception as e:
        logger.error(f"Error generating model comparison: {e}")
        return None

def generate_model_summary_table(run_ids, experiment_name, run_number=None):
    """
    Generate a summary table of model performance and log it to MLflow
    under the same experiment as the runs.

    Args:
        run_ids: List of MLflow run IDs
        experiment_name: Name of the experiment
        run_number: Run number (optional)

    Returns:
        pd.DataFrame: The summary table
    """
    try:
        import pandas as pd
        from mlflow.tracking import MlflowClient

        logger.info("Generating model performance summary table")
        client = MlflowClient()

        if run_ids:
            experiment_id = None
            try:
                first_run = client.get_run(run_ids[0])
                experiment_id = first_run.info.experiment_id
                logger.info(
                    f"Using experiment ID: {experiment_id} from run {run_ids[0]}"
                )
            except Exception as e:
                logger.warning(f"Could not get experiment ID from run: {e}")

            if experiment_id is None:
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment:
                        experiment_id = experiment.experiment_id
                        logger.info(
                            f"Using experiment ID: {experiment_id} from experiment name"
                        )
                except Exception as e:
                    logger.warning(f"Could not get experiment ID by name: {e}")
        else:
            logger.warning("No run IDs provided")
            return None

        metrics_to_include = [
            "test_r2",
            "test_rmse",
            "test_mae",
            "test_mse",
            "flops",
            "mflops",
            "num_params",
            "inference_time_ms",
            "training_time_seconds",
            "last_epoch",
            "last epoch",
        ]

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

        table_data = []
        for run in runs_data:
            run_data = {
                "run_id": run.info.run_id,
                "model": run.data.params.get("model", "unknown"),
                "dataset": run.data.params.get("dataset_type", "unknown"),
                "data_version": run.data.params.get("data_version", "unknown"),
                "status": run.data.tags.get("status", run.info.status),
            }

            for metric in metrics_to_include:
                if metric in run.data.metrics:
                    clean_metric_name = metric.replace(" ", "_")
                    run_data[clean_metric_name] = run.data.metrics[metric]

            if "epochs" in run.data.params:
                run_data["epochs"] = run.data.params["epochs"]

            for loss_type in ["train_loss", "val_loss"]:
                if f"final_{loss_type}" in run.data.metrics:
                    run_data[f"final_{loss_type}"] = run.data.metrics[
                        f"final_{loss_type}"
                    ]

            table_data.append(run_data)

        if not table_data:
            logger.warning("No data available for summary table")
            return None

        df = pd.DataFrame(table_data)

        if all(col in df.columns for col in ["dataset", "data_version", "model"]):
            df = df.sort_values(by=["dataset", "data_version", "model"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        username = os.environ.get("USER", "keirparker")

        safe_exp_name = experiment_name.replace(" ", "_").replace("/", "_")
        
        if run_number is not None:
            exp_dir = f"results/{safe_exp_name}"
            os.makedirs(exp_dir, exist_ok=True)
            
            run_dir = f"{exp_dir}/run_{run_number}"
            os.makedirs(run_dir, exist_ok=True)
            
            html_path = f"{run_dir}/summary_table.html"
            csv_path = f"{run_dir}/summary_table.csv"
        else:
            os.makedirs("results", exist_ok=True)
            
            timestr = time.strftime("%Y%m%d-%H%M%S")
            html_path = f"results/summary_table_{safe_exp_name}_{timestr}.html"
            csv_path = f"results/summary_table_{safe_exp_name}_{timestr}.csv"

        df["group"] = df["dataset"] + "_" + df["data_version"]

        metrics_to_highlight = [
            "test_r2",
            "test_rmse",
            "test_mae",
            "test_mse",
            "mflops",
            "inference_time_ms",
            "final_train_loss",
            "final_val_loss",
        ]

        highlight_cells = {metric: [] for metric in metrics_to_highlight}

        for group_name in df["group"].unique():
            group_indices = df.index[df["group"] == group_name].tolist()

            for metric in metrics_to_highlight:
                if metric in df.columns:
                    group_df = df.loc[group_indices]
                    valid_metrics = group_df[~group_df[metric].isna()]

                    if not valid_metrics.empty:
                        if metric == "test_r2":  # Higher is better
                            best_value = valid_metrics[metric].max()
                            best_indices = valid_metrics[
                                valid_metrics[metric] == best_value
                            ].index.tolist()
                        else:  # Lower is better
                            best_value = valid_metrics[metric].min()
                            best_indices = valid_metrics[
                                valid_metrics[metric] == best_value
                            ].index.tolist()

                        for idx in best_indices:
                            highlight_cells[metric].append(idx)

        html_content = []
        html_content.append("<html><head>")
        html_content.append("<style>")
        html_content.append("table { border-collapse: collapse; width: 100%; }")
        html_content.append(
            "th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }"
        )
        html_content.append("tr.group-start { border-top: 2px solid black; }")
        html_content.append("td.highlight { background-color: lightgreen; }")
        html_content.append("th { background-color: #f2f2f2; }")
        html_content.append("</style>")
        html_content.append("</head><body>")
        html_content.append(f"<h2>Model Performance Summary - {experiment_name}</h2>")
        html_content.append("<table>")

        html_content.append("<tr>")
        for col in df.columns:
            if col != "group":  # Skip the group column
                html_content.append(f"<th>{col}</th>")
        html_content.append("</tr>")

        prev_group = None
        for i, row in df.iterrows():
            if prev_group != row["group"] or i == df.index[0]:
                html_content.append('<tr class="group-start">')
            else:
                html_content.append("<tr>")

            for col in df.columns:
                if col != "group":  # Skip the group column
                    value = row[col]

                    if col in ["test_r2", "test_rmse", "test_mae", "test_mse"]:
                        if pd.notnull(value):
                            value = f"{value:.4f}"
                    elif col == "flops":
                        if pd.notnull(value):
                            value = f"{value:,.0f}"
                    elif col == "mflops":
                        if pd.notnull(value):
                            value = f"{value:.2f}"
                    elif col == "num_params":
                        if pd.notnull(value):
                            value = f"{value:,.0f}"
                    elif col == "inference_time_ms":
                        if pd.notnull(value):
                            value = f"{value:.3f}"
                    elif col in ["training_time_seconds"]:
                        if pd.notnull(value):
                            value = f"{value:.2f}"
                    elif col in ["final_train_loss", "final_val_loss"]:
                        if pd.notnull(value):
                            value = f"{value:.6f}"
                    elif col == "last_epoch":
                        if pd.notnull(value):
                            value = (
                                f"{int(value)}"
                                if isinstance(value, (int, float))
                                else value
                            )

                    if col in metrics_to_highlight and i in highlight_cells[col]:
                        html_content.append(f'<td class="highlight">{value}</td>')
                    else:
                        html_content.append(f"<td>{value}</td>")

            html_content.append("</tr>")
            prev_group = row["group"]

        html_content.append("</table>")
        html_content.append("</body></html>")

        with open(html_path, "w") as f:
            f.write("\n".join(html_content))

        df_csv = df.drop(columns=["group"])
        df_csv.to_csv(csv_path, index=False)

        try:
            if not experiment_id:
                try:
                    if run_ids:
                        first_run = client.get_run(run_ids[0])
                        experiment_id = first_run.info.experiment_id
                    
                    if not experiment_id and experiment_name:
                        experiment = mlflow.get_experiment_by_name(experiment_name)
                        if experiment:
                            experiment_id = experiment.experiment_id
                except Exception as e:
                    logger.warning(f"Could not get experiment ID: {e}")
            
            if experiment_id:
                if mlflow.active_run():
                    mlflow.end_run()
                
                timestr = time.strftime("%Y%m%d-%H%M%S")
                
                with mlflow.start_run(
                    run_name=f"Summary-{timestr}", experiment_id=experiment_id
                ):
                    mlflow.log_artifact(html_path)
                    mlflow.log_artifact(csv_path)

                    mlflow.set_tag("summary_type", "model_comparison")
                    mlflow.set_tag("table_generated_at", timestamp)
                    mlflow.set_tag("table_generated_by", username)
                    mlflow.set_tag("runs_analyzed", len(df))
                    mlflow.set_tag("is_summary", "true")
                    if run_number is not None:
                        mlflow.set_tag("run_number", str(run_number))

                    mlflow.log_param("analyzed_runs", ",".join(run_ids))

                    logger.info(
                        f"Summary table logged to MLflow experiment '{experiment_name}' and saved to {html_path}"
                    )
                    logger.info(f"Summary run ID: {mlflow.active_run().info.run_id}")
            else:
                logger.warning("Could not log to MLflow - no experiment ID found")
                logger.info(f"Summary table saved locally to {html_path}")
        except Exception as e:
            logger.error(f"Error logging summary to MLflow: {e}")
            logger.info(f"Summary table saved locally to {html_path}")

        return df_csv
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

        os.makedirs(output_dir, exist_ok=True)

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
            current_experiment = mlflow.get_experiment_by_name("Default")
            if current_experiment:
                runs = client.search_runs(experiment_ids=[current_experiment.experiment_id])
                runs_data = runs
            else:
                logger.warning("No default experiment found")

        if not runs_data:
            logger.warning("No runs found for rankings")
            return {}

        data = []
        for run in runs_data:
            run_data = {
                "Run ID": run.info.run_id,
                "Model": run.data.params.get("model", "unknown"),
                "Dataset": run.data.params.get("dataset_type", "unknown"),
                "Data Version": run.data.params.get("data_version", "unknown")
            }

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

        df = pd.DataFrame(data)

        timestr = time.strftime("%Y%m%d-%H%M%S")

        rankings = []
        model_scores = {}

        for dataset in df["Dataset"].unique():
            dataset_df = df[df["Dataset"] == dataset].copy()

            for metric in metrics:
                if metric not in dataset_df.columns:
                    continue

                dataset_metric_df = dataset_df[["Model", "Data Version", metric]].dropna().copy()

                if len(dataset_metric_df) == 0:
                    continue

                ascending = metric != "test_r2"  # True for RMSE/MAE, False for R²

                dataset_metric_df["Rank"] = dataset_metric_df[metric].rank(
                    ascending=ascending, method="min")

                dataset_metric_df["Dataset"] = dataset
                dataset_metric_df["Metric"] = metric

                if ascending:
                    max_val = dataset_metric_df[metric].max()
                    min_val = dataset_metric_df[metric].min()
                    range_val = max_val - min_val if max_val > min_val else 1
                    dataset_metric_df["Score"] = 100 * (1 - (dataset_metric_df[metric] - min_val) / range_val)
                else:
                    max_val = dataset_metric_df[metric].max()
                    min_val = dataset_metric_df[metric].min()
                    range_val = max_val - min_val if max_val > min_val else 1
                    dataset_metric_df["Score"] = 100 * (dataset_metric_df[metric] - min_val) / range_val

                for _, row in dataset_metric_df.iterrows():
                    key = (row["Model"], row["Data Version"])
                    if key not in model_scores:
                        model_scores[key] = []
                    model_scores[key].append(row["Score"])

                rankings.append(dataset_metric_df)

        if not rankings:
            logger.warning("No ranking data generated")
            return {}

        rankings_df = pd.concat(rankings)

        overall_scores = []
        for (model, data_version), scores in model_scores.items():
            overall_scores.append({
                "Model": model,
                "Data Version": data_version,
                "Overall Score": np.mean(scores),
                "Rank": 0
            })

        overall_df = pd.DataFrame(overall_scores)
        overall_df["Rank"] = overall_df["Overall Score"].rank(ascending=False, method="min")

        all_paths = {}

        rankings_path = f"{output_dir}/model_rankings_{timestr}.csv"
        rankings_df.to_csv(rankings_path, index=False)
        all_paths["rankings_csv"] = rankings_path
        logger.info(f"Model rankings saved as CSV: {rankings_path}")

        overall_path = f"{output_dir}/model_overall_scores_{timestr}.csv"
        overall_df.to_csv(overall_path, index=False)
        all_paths["overall_scores_csv"] = overall_path
        logger.info(f"Model overall scores saved as CSV: {overall_path}")
        table_df = rankings_df.pivot_table(
            index=["Model", "Data Version"],
            columns=["Dataset", "Metric"],
            values=["Rank", "Score"],
            aggfunc="first"
        )

        rankings_pivot_path = f"{output_dir}/model_rankings_pivot_{timestr}.csv"
        table_df.to_csv(rankings_pivot_path)
        all_paths["rankings_pivot_csv"] = rankings_pivot_path
        logger.info(f"Pivot table of rankings saved: {rankings_pivot_path}")

        plt.figure(figsize=(14, 10))

        try:
            rank_df = table_df.xs('Rank', axis=1, level=0)

            ax = sns.heatmap(rank_df, annot=True, cmap="YlGnBu_r", fmt=".0f",
                             linewidths=.5, cbar_kws={"label": "Rank (lower is better)"})

            plt.title("Model Rankings by Dataset and Metric", fontsize=16)
            plt.tight_layout()

            ranking_viz_path = f"{output_dir}/ranking_heatmap_{timestr}.png"
            plt.savefig(ranking_viz_path, bbox_inches='tight', dpi=300)
            plt.close()

            all_paths["ranking_heatmap"] = ranking_viz_path
            logger.info(f"Ranking heatmap saved: {ranking_viz_path}")
        except Exception as e:
            logger.error(f"Error creating ranking heatmap: {e}")

        plt.figure(figsize=(12, 8))

        plot_df = overall_df.sort_values("Overall Score", ascending=False)

        plot_df["Label"] = plot_df["Model"] + " (" + plot_df["Data Version"] + ")"

        data_versions = plot_df["Data Version"].unique()
        colors = sns.color_palette("viridis", n_colors=len(data_versions))
        color_map = {version: color for version, color in zip(data_versions, colors)}
        bar_colors = [color_map[version] for version in plot_df["Data Version"]]

        ax = sns.barplot(
            x="Overall Score",
            y="Label",
            data=plot_df,
            palette=bar_colors,
            orient="h"
        )

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

        legend_handles = [plt.Rectangle((0,0), 1, 1, color=color_map[v]) for v in data_versions]
        plt.legend(legend_handles, data_versions, title="Data Version", loc="lower right")

        overall_viz_path = f"{output_dir}/overall_scores_{timestr}.png"
        plt.savefig(overall_viz_path, bbox_inches='tight', dpi=300)
        plt.close()

        all_paths["overall_scores_viz"] = overall_viz_path
        logger.info(f"Overall scores visualization saved: {overall_viz_path}")

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
    """Generate a plot comparing learning curves across different models."""
    try:
        logger.info(f"Generating learning curve comparison for metric: {metric}")
        client = MlflowClient()

        os.makedirs(output_dir, exist_ok=True)

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

        curve_data = {}
        max_epochs = 0

        for run in runs_data:
            run_id = run.info.run_id
            model_name = run.data.params.get("model", "unknown")
            dataset = run.data.params.get("dataset_type", "unknown")
            data_version = run.data.params.get("data_version", "unknown")

            run_label = f"{model_name} - {dataset} ({data_version})"

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

        plt.figure(figsize=(14, 8))

        colors = sns.color_palette("husl", len(curve_data))

        for i, (run_label, epochs_data) in enumerate(curve_data.items()):
            epochs = list(epochs_data.keys())
            values = list(epochs_data.values())

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

        plt.xlabel("Epoch", fontsize=12)

        metric_label = metric.replace("_", " ").title()
        plt.ylabel(metric_label, fontsize=12)

        plt.title(f"Learning Curves - {metric_label}", fontsize=14)

        plt.grid(True, linestyle="--", alpha=0.7)

        if len(curve_data) > 10:
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.legend(fontsize=10)

        plt.tight_layout()

        timestr = time.strftime("%Y%m%d-%H%M%S")
        learning_curve_path = f"{output_dir}/learning_curves_{metric}_{timestr}.png"

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
    """Generate a plot comparing predictions across different models for the same dataset."""
    try:
        logger.info("Generating model prediction comparison plot")
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
    """Helper function to get run IDs from an experiment."""
    try:
        client = MlflowClient()

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

        filter_string = ""
        if filter_kwargs:
            conditions = []
            for key, value in filter_kwargs.items():
                conditions.append(f'tags."{key}" = "{value}"')
            if conditions:
                filter_string = " and ".join(conditions)

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string if filter_string else None,
        )

        run_ids = [run.info.run_id for run in runs]
        logger.info(f"Found {len(run_ids)} runs in experiment")
        return run_ids

    except Exception as e:
        logger.error(f"Error getting run IDs from experiment: {e}")
        return []


@ray.remote
def process_run_metrics(run_id, client, metric_name, include_validation=False, smooth_factor=None):
    """Ray remote function to process run metrics in parallel"""
    try:
        run = client.get_run(run_id)
        run_metrics = {
            "model_name": run.data.params.get("model", "unknown"),
            "dataset_type": run.data.params.get("dataset", run.data.params.get("dataset_type", "unknown")),
            "train_loss_by_epoch": {},
            "val_loss_by_epoch": {},
            "run": run
        }
        
        if run.data.params.get("dataset_type", "").lower() in ["electricity", "traffic", "solar-energy", "exchange-rate"]:
            if "input_dim" in run.data.params:
                input_dim = run.data.params.get("input_dim")
                run_metrics["model_name"] = f"{run_metrics['model_name']} (d={input_dim})"
        
        for key, value in run.data.metrics.items():
            if key.startswith(f"{metric_name}_epoch_") or key.startswith(f"{metric_name}"):
                try:
                    if "_epoch_" in key:
                        epoch = int(key.split("_")[-1])
                    elif key == metric_name and "epoch" in run.data.metrics:
                        epoch = int(run.data.metrics["epoch"])
                    else:
                        continue
                        
                    run_metrics["train_loss_by_epoch"][epoch] = value
                except (ValueError, TypeError):
                    continue

            if include_validation and (key.startswith("val_loss_epoch_") or key == "val_loss"):
                try:
                    if "_epoch_" in key:
                        epoch = int(key.split("_")[-1])
                    elif key == "val_loss" and "epoch" in run.data.metrics:
                        epoch = int(run.data.metrics["epoch"])
                    else:
                        continue
                        
                    run_metrics["val_loss_by_epoch"][epoch] = value
                except (ValueError, TypeError):
                    continue
        
        # Apply smoothing if requested
        if smooth_factor and len(run_metrics["train_loss_by_epoch"]) > 3:
            window_size = int(len(run_metrics["train_loss_by_epoch"]) * smooth_factor / 100)
            if window_size > 1:
                epochs = sorted(run_metrics["train_loss_by_epoch"].keys())
                losses = [run_metrics["train_loss_by_epoch"][e] for e in epochs]
                
                smoothed_losses = []
                for j in range(len(losses)):
                    start = max(0, j - window_size // 2)
                    end = min(len(losses), j + window_size // 2 + 1)
                    smoothed_losses.append(sum(losses[start:end]) / (end - start))
                
                for j, epoch in enumerate(epochs):
                    run_metrics["train_loss_by_epoch"][epoch] = smoothed_losses[j]
        
        # Apply smoothing to validation losses if requested
        if include_validation and smooth_factor and len(run_metrics["val_loss_by_epoch"]) > 3:
            window_size = int(len(run_metrics["val_loss_by_epoch"]) * smooth_factor / 100)
            if window_size > 1:
                val_epochs = sorted(run_metrics["val_loss_by_epoch"].keys())
                val_losses = [run_metrics["val_loss_by_epoch"][e] for e in val_epochs]
                
                smoothed_val_losses = []
                for j in range(len(val_losses)):
                    start = max(0, j - window_size // 2)
                    end = min(len(val_losses), j + window_size // 2 + 1)
                    smoothed_val_losses.append(sum(val_losses[start:end]) / (end - start))
                
                for j, epoch in enumerate(val_epochs):
                    run_metrics["val_loss_by_epoch"][epoch] = smoothed_val_losses[j]
        
        return run_metrics
    except Exception as e:
        logger.error(f"Error processing run {run_id}: {e}")
        return None


def plot_losses_by_epoch_comparison(
    run_ids=None,
    experiment_id=None,
    metric_name="train_loss",
    output_dir="plots",
    include_validation=True,
    smooth_factor=None,
    group_by_dataset=False,
    only_best_models=False,
    best_metric="test_rmse"
):
    """Plot loss curves by epoch for multiple runs on the same graph for comparison using Ray for parallelization."""
    try:
        logger.info(f"Generating loss comparison plot for {metric_name}")
        client = MlflowClient()

        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            try:
                # Initialize with appropriate settings for M2 Macbook
                num_cpus = os.cpu_count() or 4
                reserved_cpus = 2  # Reserve CPUs for system operations
                ray_cpus = max(1, num_cpus - reserved_cpus)
                
                ray.init(
                    num_cpus=ray_cpus,
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    _temp_dir="/tmp/ray_temp",  # Prevent permissions issues on macOS
                    _system_config={
                        "worker_register_timeout_seconds": 60,
                        "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}}',
                        "max_io_workers": 4  # Reduce I/O worker threads for Mac
                    },
                    logging_level=logging.WARNING,
                )
                logger.info(f"Ray initialized with {ray_cpus} CPUs")
            except Exception as e:
                logger.warning(f"Failed to initialize Ray: {e}. Falling back to sequential processing.")

        # Collect run IDs
        if run_ids is None and experiment_id is not None:
            runs = client.search_runs(experiment_ids=[experiment_id])
            run_ids = [run.info.run_id for run in runs]
        
        if not run_ids:
            logger.warning("No runs specified for comparison")
            return None

        # Process runs in parallel with Ray
        if ray.is_initialized():
            run_refs = [process_run_metrics.remote(
                run_id, client, metric_name, include_validation, smooth_factor
            ) for run_id in run_ids]
            
            run_metrics_list = ray.get(run_refs)
            runs_data = [rm["run"] for rm in run_metrics_list if rm is not None]
        else:
            # Fallback to sequential processing
            runs_data = []
            for run_id in run_ids:
                try:
                    run = client.get_run(run_id)
                    runs_data.append(run)
                except:
                    logger.warning(f"Could not find run with ID: {run_id}")

        if not runs_data:
            logger.warning("No runs found for comparison")
            return None
            
        if group_by_dataset:
            datasets_runs = {}
            for run in runs_data:
                dataset = run.data.params.get("dataset", run.data.params.get("dataset_type", "unknown"))
                if dataset not in datasets_runs:
                    datasets_runs[dataset] = []
                datasets_runs[dataset].append(run)
                
            if only_best_models:
                best_runs = {}
                for dataset, dataset_runs in datasets_runs.items():
                    best_score = float('inf')  # Assuming lower is better
                    best_run = None
                    
                    for run in dataset_runs:
                        if best_metric in run.data.metrics:
                            score = run.data.metrics[best_metric]
                            if score < best_score:  # Better score found
                                best_score = score
                                best_run = run
                    
                    if best_run:
                        best_runs[dataset] = [best_run]
                    else:
                        best_runs[dataset] = dataset_runs
                        
                datasets_runs = best_runs
                
            plot_paths = {}
            for dataset, dataset_runs in datasets_runs.items():
                plot_path = _create_loss_plot(
                    dataset_runs, 
                    metric_name, 
                    output_dir, 
                    include_validation,
                    smooth_factor,
                    dataset_name=dataset
                )
                if plot_path:
                    plot_paths[dataset] = plot_path
                    
            return plot_paths
        else:
            return _create_loss_plot(
                runs_data, 
                metric_name, 
                output_dir, 
                include_validation,
                smooth_factor
            )
    except Exception as e:
        logger.error(f"Error generating loss comparison plot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
            
def _create_loss_plot(runs_data, metric_name, output_dir, include_validation, smooth_factor, dataset_name=None):
    """Helper function to create a loss plot for a set of runs."""
    try:
        plt.figure(figsize=(12, 8))

        colors = sns.color_palette("husl", len(runs_data))

        max_epochs = 0

        for i, run in enumerate(runs_data):
            model_name = run.data.params.get("model", "unknown")
            dataset_type = run.data.params.get("dataset", run.data.params.get("dataset_type", "unknown"))
            
            if dataset_type.lower() in ["electricity", "traffic", "solar-energy", "exchange-rate"]:
                if "input_dim" in run.data.params:
                    input_dim = run.data.params.get("input_dim")
                    model_name = f"{model_name} (d={input_dim})"

            run_name = model_name

            train_loss_by_epoch = {}
            val_loss_by_epoch = {}

            for key, value in run.data.metrics.items():
                if key.startswith(f"{metric_name}_epoch_") or key.startswith(f"{metric_name}"):
                    try:
                        if "_epoch_" in key:
                            epoch = int(key.split("_")[-1])
                        elif key == metric_name and "epoch" in run.data.metrics:
                            epoch = int(run.data.metrics["epoch"])
                        else:
                            continue
                            
                        train_loss_by_epoch[epoch] = value
                        max_epochs = max(max_epochs, epoch)
                    except (ValueError, TypeError):
                        continue

                if include_validation and (key.startswith("val_loss_epoch_") or key == "val_loss"):
                    try:
                        if "_epoch_" in key:
                            epoch = int(key.split("_")[-1])
                        elif key == "val_loss" and "epoch" in run.data.metrics:
                            epoch = int(run.data.metrics["epoch"])
                        else:
                            continue
                            
                        val_loss_by_epoch[epoch] = value
                        max_epochs = max(max_epochs, epoch)
                    except (ValueError, TypeError):
                        continue

            if smooth_factor and len(train_loss_by_epoch) > 3:
                window_size = int(len(train_loss_by_epoch) * smooth_factor / 100)
                if window_size > 1:
                    epochs = sorted(train_loss_by_epoch.keys())
                    losses = [train_loss_by_epoch[e] for e in epochs]
                    
                    smoothed_losses = []
                    for j in range(len(losses)):
                        start = max(0, j - window_size // 2)
                        end = min(len(losses), j + window_size // 2 + 1)
                        smoothed_losses.append(sum(losses[start:end]) / (end - start))
                    
                    for j, epoch in enumerate(epochs):
                        train_loss_by_epoch[epoch] = smoothed_losses[j]

            if train_loss_by_epoch:
                epochs = sorted(train_loss_by_epoch.keys())
                losses = [train_loss_by_epoch[e] for e in epochs]

                line = plt.plot(
                    epochs,
                    losses,
                    label=f"{run_name} (train)",
                    color=colors[i],
                    linewidth=1.5,
                    alpha=0.8,
                )
                
                plt.scatter(
                    epochs,
                    losses,
                    marker='o',
                    s=40,
                    color=colors[i],
                    alpha=0.9,
                    edgecolors='white',
                    linewidth=0.7,
                    zorder=10,
                )

            if include_validation and val_loss_by_epoch:
                val_epochs = sorted(val_loss_by_epoch.keys())
                val_losses = [val_loss_by_epoch[e] for e in val_epochs]

                if smooth_factor and len(val_loss_by_epoch) > 3:
                    window_size = int(len(val_loss_by_epoch) * smooth_factor / 100)
                    if window_size > 1:
                        smoothed_val_losses = []
                        for j in range(len(val_losses)):
                            start = max(0, j - window_size // 2)
                            end = min(len(val_losses), j + window_size // 2 + 1)
                            smoothed_val_losses.append(sum(val_losses[start:end]) / (end - start))
                        
                        val_losses = smoothed_val_losses

                line_val = plt.plot(
                    val_epochs,
                    val_losses,
                    label=f"{run_name} (val)",
                    color=colors[i],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )
                
                plt.scatter(
                    val_epochs,
                    val_losses,
                    marker='x',
                    s=40,
                    color=colors[i],
                    alpha=0.9,
                    edgecolors='white',
                    linewidth=0.8,
                    zorder=10,
                )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        
        if dataset_name:
            title = f"{dataset_name} - Loss Comparison Across Models"
        else:
            title = "Loss Comparison Across Models by Epoch"
            
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        
        if len(runs_data) > 5:
            plt.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
        else:
            plt.legend(fontsize=10, loc="best")

        plt.tight_layout()

        timestr = time.strftime("%Y%m%d-%H%M%S")

        if dataset_name:
            dataset_str = dataset_name.replace(" ", "_").lower()
            comparison_path = f"{output_dir}/loss_comparison_{dataset_str}_{len(runs_data)}_models_{timestr}.png"
        elif len(runs_data) > 3:
            comparison_path = f"{output_dir}/loss_comparison_{len(runs_data)}_models_{timestr}.png"
        else:
            model_names = [run.data.params.get("model", "unknown") for run in runs_data]
            model_str = "_".join(model_names)
            comparison_path = f"{output_dir}/loss_comparison_{model_str}_{timestr}.png"

        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(comparison_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Loss comparison plot saved to {comparison_path}")
        return comparison_path
        
    except Exception as e:
        logger.error(f"Error creating loss plot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None