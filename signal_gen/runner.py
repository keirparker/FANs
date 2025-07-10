#!/usr/bin/env python
"""Main entry point for the signal generation and processing framework."""

import mlflow
from loguru import logger
import logging
import os
import time
from datetime import datetime
import json
import numpy as np
import platform
import ray
from ray.air import session
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from functools import partial
import psutil
import sys

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


def init_ray_for_mac():
    num_cpus = psutil.cpu_count(logical=True)
    reserved_cpus = 2
    ray_cpus = max(1, num_cpus - reserved_cpus)
    
    if not ray.is_initialized():
        try:
            ray.init(
                num_cpus=ray_cpus,
                include_dashboard=False,  # Disable dashboard for stability
                ignore_reinit_error=True,
                _temp_dir="/tmp/ray_temp",  # Prevent permissions issues on macOS
                _system_config={
                    "worker_register_timeout_seconds": 60,
                    # Removing problematic parameter: "raylet_startup_token_refresh_ms"
                    "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}}',
                    "max_io_workers": 4  # Reduce I/O worker threads for Mac
                },
                logging_level=logging.WARNING
            )
            mac_info = platform.mac_ver()
            logger.info(f"Ray initialized with {ray_cpus} CPUs on macOS {mac_info[0]} {platform.processor()}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}. Cannot continue with parallel processing.")
            return False


def run_experiment_sequential(model_name, dataset_type, data_version, config, experiment_id, experiment_name=None, run_number=None):
    """Run a single experiment in sequential mode (without Ray)."""
    return _run_experiment_impl(model_name, dataset_type, data_version, config, experiment_id, experiment_name, run_number)


def _run_experiment_impl(model_name, dataset_type, data_version, config, experiment_id, experiment_name=None, run_number=None):
    """Common implementation for both Ray and sequential experiments."""
    run_id = None
    try:
        try:
            model_test = get_model_by_name(model_name)
            logger.info(f"Successfully verified model {model_name} exists")
        except ValueError as e:
            logger.error(f"Cannot run experiment with model '{model_name}': {e}")
            from signal_gen.src.models import model_registry
            logger.error(f"Available models: {list(model_registry.keys())}")
            return None

        if experiment_name is None:
            experiment_name = f"experiment_{model_name}_{dataset_type}_{data_version}"
        
        if run_number is not None:
            run_name = f"{experiment_name}_run_{run_number}"
        else:
            run_name = experiment_name

        logger.info(f"Starting experiment: {run_name}")
        
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            
            # Log experiment parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_type", dataset_type)
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("run_number", run_number)
            
            # Log hyperparameters
            for key, value in config["hyperparameters"].items():
                mlflow.log_param(key, value)
            
            # Set up device
            device, device_info = select_device(config)
            mlflow.log_param("device", device.type)
            mlflow.log_param("device_info", device_info)
            
            # Generate or load data
            original_data = get_periodic_data(dataset_type, config)
            
            # Apply data version modifications
            if data_version == "original":
                data = original_data
            elif data_version == "noisy":
                data = add_noise(original_data, config)
            elif data_version == "sparse":
                data = make_sparse(original_data, config)
            else:
                logger.error(f"Unknown data version: {data_version}")
                return None
            
            # Create and train model
            model = get_model_by_name(model_name)(
                input_dim=data.shape[1] if len(data.shape) > 1 else 1,
                output_dim=data.shape[1] if len(data.shape) > 1 else 1,
                **config["hyperparameters"]
            )
            
            model = model.to(device)
            
            # Log model parameters
            total_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("total_parameters", total_params)
            
            # Train the model
            optimizer = create_optimizer(model, config)
            scheduler = create_scheduler(optimizer, config)
            
            model, train_losses, val_losses = train_model(
                model, data, device, config, run_id, optimizer, scheduler
            )
            
            # Evaluate the model
            metrics = evaluate_model(model, data, device, config, run_id)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Generate and log visualizations
            try:
                plot_training_history(train_losses, val_losses, run_id)
                plot_model_predictions(model, data, device, run_id)
                log_plots_to_mlflow(run_id)
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {e}")
            
            logger.info(f"Experiment completed: {run_name}, run_id: {run_id}")
            return run_id
            
    except Exception as e:
        logger.error(f"Experiment failed for {model_name}_{dataset_type}_{data_version}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


@ray.remote
def run_experiment_ray(model_name, dataset_type, data_version, config, experiment_id, experiment_name=None, run_number=None):
    """Run a single experiment using Ray remote execution."""
    return _run_experiment_impl(model_name, dataset_type, data_version, config, experiment_id, experiment_name, run_number)

def save_run_ids(run_ids, experiment_name, run_number=None):
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
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            logger.info("Apple Silicon MPS available")
            
            device = torch.device("mps")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            start_time = time.time()
            z = torch.matmul(x, y)
            duration = time.time() - start_time
            
            logger.info(f"MPS test successful! Result shape: {z.shape}")
            logger.info(f"Operation completed in {duration*1000:.2f} ms")
            
            del x, y, z
            
            logger.info("MPS test passed ✓")
        except Exception as e:
            logger.error(f"MPS test failed: {e}")
    else:
        logger.warning("CUDA/MPS is not available. Will run on CPU instead.")
    
    logger.info("================================")

def main():
    setup_logger()
    
    test_gpu_functionality()
    
    # Load config first to check Ray settings
    config = load_config()
    
    # Check Ray configuration
    ray_config = config.get("ray_config", {})
    fail_on_ray_error = ray_config.get("fail_on_ray_error", True)
    force_sequential = ray_config.get("force_sequential", False)
    
    if force_sequential:
        logger.info("Ray disabled by configuration (force_sequential=true)")
        ray_available = False
    else:
        ray_available = init_ray_for_mac()
        
        if not ray_available:
            if fail_on_ray_error:
                logger.error("Ray initialization failed on M2 MacBook Air!")
                logger.error("This will cause experiments to run sequentially and may produce inconsistent results.")
                logger.error("To fix this: 1) Check Ray installation, 2) Reduce num_cpus in init_ray_for_mac(), or 3) Set fail_on_ray_error=false in config.yml")
                raise RuntimeError("Ray initialization failed - aborting to prevent incorrect results")
            else:
                logger.warning("Ray initialization failed. Continuing with sequential mode (fail_on_ray_error=false).")
                logger.warning("CAUTION: Sequential mode may produce different results and will be significantly slower.")

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

        all_experiments = []
        for model_name in models_to_run:
            for dataset_type in dataset_types:
                for data_version in data_versions:
                    all_experiments.append((model_name, dataset_type, data_version))

        # Run experiments based on Ray availability
        if ray_available:
            logger.info("Running experiments in parallel using Ray")
            experiment_refs = []
            for model_name, dataset_type, data_version in all_experiments:
                logger.info(
                    f"Scheduling experiment: model={model_name}, dataset={dataset_type}, version={data_version}"
                )
                ref = run_experiment_ray.remote(
                    model_name,
                    dataset_type,
                    data_version,
                    config,
                    experiment_id,
                    experiment_name=experiment_name,
                    run_number=run_number
                )
                experiment_refs.append(ref)

            if experiment_refs:
                logger.info(f"Waiting for {len(experiment_refs)} experiments to complete...")
                try:
                    import tqdm
                    all_run_ids = []
                    for i, ref in enumerate(experiment_refs):
                        try:
                            run_id = ray.get(ref)
                            if run_id:
                                all_run_ids.append(run_id)
                            logger.info(f"Completed experiment {i+1}/{len(experiment_refs)}")
                        except Exception as e:
                            logger.error(f"Ray experiment {i+1} failed: {e}")
                            if fail_on_ray_error:
                                raise RuntimeError(f"Ray experiment execution failed: {e}")
                except ImportError:
                    try:
                        all_run_ids = ray.get(experiment_refs)
                        all_run_ids = [run_id for run_id in all_run_ids if run_id]
                    except Exception as e:
                        logger.error(f"Ray batch execution failed: {e}")
                        if fail_on_ray_error:
                            raise RuntimeError(f"Ray batch execution failed: {e}")
                        all_run_ids = []
                
                logger.info(f"All experiments completed, {len(all_run_ids)} successful runs")
            else:
                logger.warning("No experiments were scheduled")
                all_run_ids = []
        else:
            logger.info("Running experiments sequentially (Ray not available)")
            all_run_ids = []
            for model_name, dataset_type, data_version in all_experiments:
                logger.info(
                    f"Running sequential experiment: model={model_name}, dataset={dataset_type}, version={data_version}"
                )
                
                # Run experiment directly without Ray
                run_id = run_experiment_sequential(
                    model_name,
                    dataset_type,
                    data_version,
                    config,
                    experiment_id,
                    experiment_name=experiment_name,
                    run_number=run_number
                )
                if run_id:
                    all_run_ids.append(run_id)
                    
            logger.info(f"All sequential experiments completed, {len(all_run_ids)} successful runs")

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