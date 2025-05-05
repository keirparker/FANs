#!/usr/bin/env python
"""Main entry point for the signal generation and processing framework."""

import mlflow
from loguru import logger
import os
import time
from datetime import datetime
import json
import numpy as np
import platform

import sys
# Add the parent directory to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signal_gen.utils.config_utils import load_config, setup_environment
from signal_gen.utils.device_utils import select_device
from signal_gen.utils.data_utils import add_noise, make_sparse
from signal_gen.utils.training_utils import train_model, create_optimizer, create_scheduler
from signal_gen.utils.modelling_utils import evaluate_model
from signal_gen.utils.evaluation_utils import (
    generate_model_summary_table,
    plot_losses_by_epoch_comparison
)
from signal_gen.utils.visualisation_utils import (
    plot_training_history,
    plot_model_predictions,
    log_plots_to_mlflow,
    create_enhanced_visualizations
)
from signal_gen.src.signal_gen import get_periodic_data
from signal_gen.src.models import get_model_by_name


def setup_logger():
    """Configure the logger."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(
        f"{log_dir}/experiments_{time.strftime('%Y%m%d-%H%M%S')}.log",
        rotation="50 MB",
        retention="5 days",
        compression="zip",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(
        lambda msg: print(msg),
        level="INFO",
        format="{level} | <blue>{time:HH:mm:ss}</blue> | <level>{message}</level>",
    )


def run_experiment(model_name, dataset_type, data_version, config, experiment_id, experiment_name=None, run_number=None):
    """Runs a single experiment with the given configuration."""
    run_id = None
    try:
        # First check if the model exists before starting a run
        try:
            model_test = get_model_by_name(model_name)
            logger.info(f"Successfully verified model {model_name} exists")
        except ValueError as e:
            logger.error(f"Cannot run experiment with model '{model_name}': {e}")
            # Get list of available models
            from signal_gen.src.models import model_registry
            available_models = sorted(list(model_registry.keys()))
            logger.info(f"Available models: {available_models}")
            logger.info("Please update your config.yml file to use one of the available models")
            return None
            
        run_name = f"{model_name}_{dataset_type}_{data_version}"

        try:
            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                logger.info(f"Started run {run_id} in experiment {experiment_id}")

                mlflow.log_param("model", model_name)
                mlflow.log_param("dataset_type", dataset_type)
                mlflow.log_param("data_version", data_version)
                mlflow.log_param("num_samples", config["hyperparameters"]["num_samples"])
                mlflow.log_param("epochs", config["hyperparameters"]["epochs"])
                mlflow.log_param("lr", config["hyperparameters"]["lr"])

                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("dataset", dataset_type)
                mlflow.set_tag("data_version", data_version)
                mlflow.set_tag("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                mlflow.set_tag("user", os.environ.get("USER", "keirparker"))

                device, device_info = select_device(config)
                mlflow.log_param("device", device.type)
                
                mlflow.log_param("device_name", device_info.get('name', 'Unknown'))
                
                if device_info.get('is_multi_gpu', False):
                    mlflow.log_param("gpu_count", device_info.get('gpu_count', 0))
                    mlflow.log_param("multi_gpu", True)
                
                if device_info.get('aws_instance', None):
                    mlflow.log_param("aws_instance", device_info['aws_instance'])
                
                if device_info.get('apple_silicon', False):
                    mlflow.log_param("apple_silicon", True)
                    if device_info.get('chip_model'):
                        mlflow.log_param("chip_model", device_info['chip_model'])
                        
                if device_info.get('is_windows', False):
                    mlflow.log_param("windows", True)
                    mlflow.log_param("windows_version", platform.release())
                    
                    if device_info.get('is_windows_gigabyte', False):
                        mlflow.log_param("windows_gigabyte_gpu", True)
                        
                        gigabyte_info = device_info.get('gigabyte_gpu_info', {})
                        if gigabyte_info:
                            if gigabyte_info.get('model'):
                                mlflow.log_param("gigabyte_gpu_model", gigabyte_info.get('model'))
                            if gigabyte_info.get('vram'):
                                mlflow.log_param("gigabyte_gpu_vram", f"{gigabyte_info.get('vram')} MB")
                            if gigabyte_info.get('driver_version'):
                                mlflow.log_param("nvidia_driver", gigabyte_info.get('driver_version'))

                logger.info(f"Generating {dataset_type} dataset...")
                (t_train, data_train, t_test, data_test, data_config, true_func) = (
                    get_periodic_data(
                        periodic_type=dataset_type,
                        num_train_samples=config["hyperparameters"]["num_samples"],
                        num_test_samples=config["hyperparameters"]["test_samples"],
                    )
                )

                experiment_batch = config.get("experiment_name", "FAN_Model_Benchmark")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_id = f"{experiment_batch}_{timestamp}"
                mlflow.set_tag("batch_id", batch_id)
                mlflow.set_tag("experiment_batch", experiment_batch)

                logger.info(f"Generating {dataset_type} dataset...")
                (t_train, data_train, t_test, data_test, data_config, true_func) = (
                    get_periodic_data(
                        periodic_type=dataset_type,
                        num_train_samples=config["hyperparameters"]["num_samples"],
                        num_test_samples=config["hyperparameters"]["test_samples"],
                    )
                )

                mlflow.log_metric("train_size", len(t_train))
                mlflow.log_metric("test_size", len(t_test))

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
                    t_train = t_train[idx_train]
                    t_test = t_test[idx_test]
                    mlflow.log_param("sparsity_factor", sparsity_factor)
                    mlflow.log_metric("train_size_after_sparse", len(t_train))
                    mlflow.log_metric("test_size_after_sparse", len(t_test))
                else:
                    logger.error(f"Unknown data version: {data_version}")
                    raise ValueError("Invalid data version")

                validation_split = config["hyperparameters"].get("validation_split", 0.1)
                mlflow.log_param("validation_split", validation_split)

                logger.info(f"Initializing model: {model_name}")
                model = get_model_by_name(model_name)
                model.to(device)

                optimizer = create_optimizer(model, config)
                scheduler = create_scheduler(optimizer, config)
                
                resume_training = config.get("resume_training", False)
                start_epoch = 0
                checkpoint_dir = os.path.join("models", "checkpoints")
                
                # Enhanced checkpoint loading with better feedback
                if resume_training:
                    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                    
                    if os.path.exists(best_model_path):
                        checkpoint_path = best_model_path
                        logger.info("Found best model checkpoint, attempting to resume training")
                    else:
                        # Look for any checkpoints
                        if os.path.exists(checkpoint_dir):
                            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
                            if checkpoint_files:
                                latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
                                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                                logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                            else:
                                checkpoint_path = None
                                logger.info("No checkpoints found, starting training from scratch")
                        else:
                            # Create checkpoint directory if it doesn't exist
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            checkpoint_path = None
                            logger.info("No checkpoints directory found, starting training from scratch")
                    
                    if checkpoint_path:
                        from signal_gen.utils.training_utils import load_checkpoint
                        start_epoch, loaded_history, loaded_config = load_checkpoint(
                            model, optimizer, scheduler, checkpoint_path
                        )
                        
                        if start_epoch > 0 and loaded_history:
                            logger.info(f"Resuming training from epoch {start_epoch}")
                            
                            # Show validation loss from checkpoint
                            if loaded_history and 'val_loss' in loaded_history and loaded_history['val_loss']:
                                last_val_loss = loaded_history['val_loss'][-1]
                                best_val_loss = min(loaded_history['val_loss'])
                                logger.info(f"Previous training reached validation loss of {last_val_loss:.6f} (best: {best_val_loss:.6f})")
                            
                            # Merge any missing configuration
                            if loaded_config:
                                for k, v in loaded_config.items():
                                    if k not in config:
                                        config[k] = v
                                        logger.info(f"Loaded missing config parameter from checkpoint: {k}")
                        else:
                            logger.warning("Failed to load checkpoint, starting from scratch")
                            start_epoch = 0

                train_start_time = time.time()

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

                training_time = time.time() - train_start_time
                mlflow.log_metric("training_time_seconds", training_time)
                mlflow.log_metric("last epoch", history["epochs"][-1])
                logger.info(f"Training completed in {training_time:.2f} seconds")

                logger.info("Evaluating model on test data")
                try:
                    test_metrics, predictions = evaluate_model(model, t_test, data_test, device)
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    test_metrics = {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "r2": float('nan')}
                    predictions = np.zeros_like(data_test)

                try:
                    safe_metrics = {}
                    for metric_name, metric_value in test_metrics.items():
                        if isinstance(metric_value, (int, float)) and (np.isnan(metric_value) or np.isinf(metric_value)):
                            logger.warning(f"Found NaN/Inf in metric {metric_name}, replacing with fallback value")
                            if metric_name == "r2":
                                safe_metrics[metric_name] = -1.0
                            else:
                                safe_metrics[metric_name] = 99.0
                        else:
                            safe_metrics[metric_name] = metric_value
                    
                    mlflow.log_metric("test_mse", safe_metrics["mse"])
                    mlflow.log_metric("test_rmse", safe_metrics["rmse"])
                    mlflow.log_metric("test_mae", safe_metrics["mae"])
                    mlflow.log_metric("test_r2", safe_metrics["r2"])
                except Exception as e:
                    logger.error(f"Error logging metrics: {e}")
                    mlflow.log_metric("test_mse", 999999)
                    mlflow.log_metric("test_rmse", 999999)
                    mlflow.log_metric("test_mae", 999999)
                    mlflow.log_metric("test_r2", -999999)
                
                if "flops" in test_metrics:
                    mlflow.log_metric("flops", test_metrics["flops"])
                    mlflow.log_metric("mflops", test_metrics["mflops"])
                    mlflow.log_metric("num_params", test_metrics["num_params"])
                    mlflow.log_metric("inference_time_ms", test_metrics["inference_time_ms"])

                if history["train_loss"]:
                    mlflow.log_metric("final_train_loss", history["train_loss"][-1])
                    mlflow.log_metric("min_train_loss", min(history["train_loss"]))

                    epochs = len(history["train_loss"])
                    if epochs <= 10:
                        for i, loss in enumerate(history["train_loss"]):
                            mlflow.log_metric(f"train_loss_epoch_{i + 1}", loss)
                    else:
                        mlflow.log_metric("train_loss_epoch_1", history["train_loss"][0])
                        mlflow.log_metric(
                            f"train_loss_epoch_{epochs}", history["train_loss"][-1]
                        )
                        min_loss_idx = history["train_loss"].index(min(history["train_loss"]))
                        mlflow.log_metric(
                            f"train_loss_epoch_{min_loss_idx + 1}",
                            history["train_loss"][min_loss_idx],
                        )
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

                if history["val_loss"] and len(history["val_loss"]) > 0:
                    mlflow.log_metric("final_val_loss", history["val_loss"][-1])
                    mlflow.log_metric("min_val_loss", min(history["val_loss"]))

                    epochs = len(history["val_loss"])
                    if epochs <= 10:
                        for i, loss in enumerate(history["val_loss"]):
                            mlflow.log_metric(f"val_loss_epoch_{i + 1}", loss)
                    else:
                        mlflow.log_metric("val_loss_epoch_1", history["val_loss"][0])
                        mlflow.log_metric(
                            f"val_loss_epoch_{epochs}", history["val_loss"][-1]
                        )
                        min_loss_idx = history["val_loss"].index(min(history["val_loss"]))
                        mlflow.log_metric(
                            f"val_loss_epoch_{min_loss_idx + 1}",
                            history["val_loss"][min_loss_idx],
                        )
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

                if "metrics" in history and history["metrics"]:
                    if history["metrics"] and len(history["metrics"]) > 0:
                        for metric_name in history["metrics"][0].keys():
                            metric_values = [m.get(metric_name, None) for m in history["metrics"]]
                            metric_values = [v for v in metric_values if v is not None]

                            if metric_values:
                                mlflow.log_metric(f"final_{metric_name}", metric_values[-1])

                                if metric_name == "r2":
                                    best_value = max(metric_values)
                                    best_epoch = metric_values.index(best_value) + 1
                                    mlflow.log_metric(f"best_{metric_name}", best_value)
                                    mlflow.log_metric(f"best_{metric_name}_epoch", best_epoch)
                                else:
                                    best_value = min(metric_values)
                                    best_epoch = metric_values.index(best_value) + 1
                                    mlflow.log_metric(f"best_{metric_name}", best_value)
                                    mlflow.log_metric(f"best_{metric_name}_epoch", best_epoch)

                history_plots = plot_training_history(
                    history, 
                    model_name, 
                    dataset_type, 
                    data_version,
                    experiment_name=experiment_name,
                    run_number=run_number
                )
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
                log_plots_to_mlflow(history_plots + [prediction_plot])

                if config.get("save_model", False):
                    model_dir = os.path.join("models", f"{model_name}_{dataset_type}_{data_version}")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, "model.pt")
                    mlflow.log_artifact(model_path)

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
    """Save run IDs to a JSON file for later analysis."""
    if not run_ids:
        logger.warning("No run IDs to save")
        return

    safe_exp_name = experiment_name.replace(" ", "_").replace("/", "_")
    
    if run_number is not None:
        exp_dir = f"results/{safe_exp_name}"
        os.makedirs(exp_dir, exist_ok=True)
        
        run_dir = f"{exp_dir}/run_{run_number}"
        os.makedirs(run_dir, exist_ok=True)
        
        filename = f"{run_dir}/run_ids.json"
    else:
        os.makedirs("results", exist_ok=True)
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"results/run_ids_{safe_exp_name}_{timestr}.json"

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
    """Run a quick test to verify if the GPU is properly detected and functioning."""
    logger.info("=== GPU/CUDA DETECTION TEST ===")
    import torch
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Device count: {torch.cuda.device_count()}")
        
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Device name: {device_name}")
        
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            logger.info(f"VRAM: {total_mem / (1024**3):.2f} GB total, {free_mem / (1024**3):.2f} GB free")
        except:
            logger.info("Could not query GPU memory info")
        
        try:
            logger.info("Running GPU test tensor operation...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            
            start_time = time.time()
            
            z = torch.matmul(x, y)
            
            duration = time.time() - start_time
            
            torch.cuda.synchronize()
            
            logger.info(f"GPU test successful! Result shape: {z.shape}")
            logger.info(f"Operation completed in {duration*1000:.2f} ms")
            
            del x, y, z
            torch.cuda.empty_cache()
            
            logger.info("GPU test passed ✓")
        except Exception as e:
            logger.error(f"GPU test failed: {e}")
    else:
        logger.warning("CUDA is not available. Will run on CPU instead.")
    
    logger.info("================================")

def main():
    """Main entry point. Loads configuration and runs all specified experiments."""
    setup_logger()
    
    test_gpu_functionality()

    mlflow_dir = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
    if os.path.exists(mlflow_dir):
        try:
            experiment_dirs = [d for d in os.listdir(mlflow_dir) 
                              if d.isdigit() or (d.replace("-", "").isdigit() and d != "0")]
            if len(experiment_dirs) > 3:
                experiment_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(mlflow_dir, x)))
                for old_dir in experiment_dirs[:-3]:
                    old_path = os.path.join(mlflow_dir, old_dir)
                    logger.info(f"Cleaning up old MLflow data: {old_path}")
                    import shutil
                    shutil.rmtree(old_path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error cleaning up MLflow directories: {e}")
    
    log_dir = "logs"
    if os.path.exists(log_dir):
        try:
            log_files = [f for f in os.listdir(log_dir) if f.endswith(".log") and not f.endswith(".zip")]
            if len(log_files) > 10:
                log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
                for old_log in log_files[:-10]:
                    old_log_path = os.path.join(log_dir, old_log)
                    logger.info(f"Removing old log file: {old_log_path}")
                    os.remove(old_log_path)
        except Exception as e:
            logger.warning(f"Error cleaning up log files: {e}")

    logger.info(
        f"ML experiment framework started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Running as user: {os.environ.get('USER', 'keirparker')}")

    try:
        config_path = os.path.join(os.path.dirname(__file__), "configs/config.yml")
        config = load_config(config_path)
        setup_environment(config)

        models_to_run = config["models_to_run"]
        dataset_types = config["datasets_to_run"]
        data_versions = config["data_versions"]

        total_experiments = len(models_to_run) * len(dataset_types) * len(data_versions)
        logger.info(f"Starting {total_experiments} experiments")

        experiment_name = config.get("experiment_name", "FAN_Model_Benchmark")
        
        run_number = config.get("run_number")
        if run_number is None:
            safe_exp_name = experiment_name.replace(" ", "_").replace("/", "_")
            exp_dir = f"results/{safe_exp_name}"
            if os.path.exists(exp_dir):
                existing_runs = [d for d in os.listdir(exp_dir) if d.startswith("run_")]
                if existing_runs:
                    existing_numbers = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
                    run_number = max(existing_numbers) + 1 if existing_numbers else 1
                else:
                    run_number = 1
            else:
                run_number = 1
            
            logger.info(f"Automatically assigned run number: {run_number}")

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
                experiment_id = "0"
                logger.warning("Using default experiment (ID: 0)")
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing experiment: {experiment_name} (ID: {experiment_id})"
            )

        all_run_ids = []

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

        if not all_run_ids:
            logger.warning("No successful runs were completed. Check the config.yml file.")
            logger.info("Listing available models:")
            from signal_gen.src.models import model_registry
            available_models = sorted(list(model_registry.keys()))
            logger.info(f"Available models: {available_models}")
            return
            
        save_run_ids(all_run_ids, experiment_name, run_number)

        logger.info("Generating summary table...")
        summary_df = generate_model_summary_table(
            all_run_ids, 
            experiment_name, 
            run_number=run_number
        )
        if summary_df is not None:
            logger.info(f"Summary table generated with {len(summary_df)} runs")
        else:
            logger.warning("Failed to generate summary table")

        logger.info("Generating loss comparison plots...")

        train_loss_plot = plot_losses_by_epoch_comparison(
            run_ids=all_run_ids, 
            metric_name="train_loss", 
            include_validation=True,
            smooth_factor=None
        )

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
        if mlflow.active_run():
            mlflow.end_run()

        logger.info(
            f"ML experiment framework finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )



if __name__ == "__main__":
    main()