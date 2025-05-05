import mlflow
import mlflow.pytorch
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.signal_gen import get_periodic_data
from utils.data_utils import add_noise, make_sparse
from src.models import get_model_by_name
from utils.efficiency_utils import count_params, count_flops, measure_inference_time


from utils.device_utils import select_device as device_utils_select_device

def select_device(config):
    """Determine which device to use based on config and hardware availability."""
    return device_utils_select_device(config)


def train_model(model, t_train, data_train, config, device):
    """
    Minimal example training loop with device support.
    Args:
        model: A PyTorch model instance.
        t_train: np.ndarray of shape (N,) with training 'x' values.
        data_train: np.ndarray of shape (N,) with training 'y' values.
        config: Dictionary of hyperparameters.
        device: torch.device to run on (cpu, cuda, or mps).

    Returns:
        final_train_loss (float): The final training loss after the last epoch.
    """
    x_tensor = torch.from_numpy(t_train).float().unsqueeze(-1).to(device)
    y_tensor = torch.from_numpy(data_train).float().unsqueeze(-1).to(device)

    criterion = nn.MSELoss()
    lr = config["hyperparameters"].get("lr", 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = config["hyperparameters"].get("epochs", 10)
    batch_size = min(64, len(x_tensor))
    
    x_tensor_cpu = x_tensor.cpu()
    y_tensor_cpu = y_tensor.cpu()
    
    dataset = torch.utils.data.TensorDataset(x_tensor_cpu, y_tensor_cpu)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=False
    )

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    return epoch_loss


def evaluate_model(model, t_test, data_test, device):
    """
    Evaluate the model on test data, returning evaluation metrics and predictions.

    Args:
        model: A PyTorch model instance.
        t_test: np.ndarray of shape (N,) with test 'x' values.
        data_test: np.ndarray of shape (N,) with test 'y' values.
        device: torch.device to run on (cpu, cuda, or mps).

    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
        predictions (np.ndarray): Model predictions, shape (N,).
    """
    try:
        model_copy = copy.deepcopy(model)
    except Exception:
        logger.warning("Could not create deep copy of model, using original model")
        model_copy = model
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.metrics import (
        mean_squared_error,
        r2_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
    )

    criterion = nn.MSELoss()

    x_tensor = torch.from_numpy(t_test).float().unsqueeze(-1).to(device)
    y_tensor = torch.from_numpy(data_test).float().unsqueeze(-1).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(x_tensor)
        mse_loss = criterion(preds, y_tensor).item()

    preds_np = preds.squeeze(-1).cpu().numpy()
    
    if np.isnan(preds_np).any():
        logger.warning(f"NaN values detected in predictions: {np.sum(np.isnan(preds_np))} NaNs")
        preds_np = np.nan_to_num(preds_np, nan=0.0)
        
    if np.isnan(data_test).any():
        logger.warning(f"NaN values detected in test data: {np.sum(np.isnan(data_test))} NaNs")
        data_test = np.nan_to_num(data_test, nan=0.0)
        
    mse = mean_squared_error(data_test, preds_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(data_test, preds_np)
    r2 = r2_score(data_test, preds_np)

    try:
        if hasattr(model_copy, "count_params") and callable(getattr(model_copy, "count_params")):
            num_params = model_copy.count_params()
        else:
            num_params = count_params(model_copy)
        
        metrics = {
            "mse": mse, 
            "rmse": rmse, 
            "mae": mae, 
            "r2": r2,
            "num_params": num_params,
        }
        
        if any(model_name in model_copy.__class__.__name__ for model_name in 
               ["FANPhaseOffsetModelGated", "FANPhaseOffsetModelUniform", "FANPhaseOffsetModel"]):
            try:
                if hasattr(model_copy, "get_flops") and callable(getattr(model_copy, "get_flops")):
                    flops = model_copy.get_flops()
                else:
                    flops = num_params * 3
                    
                if hasattr(model_copy, "measure_inference_time") and callable(getattr(model_copy, "measure_inference_time")):
                    inference_time = model_copy.measure_inference_time()
                else:
                    inference_time = num_params / 1e6
                
                if math.isnan(flops) or flops <= 0:
                    flops = num_params * 3
                if math.isnan(inference_time) or inference_time <= 0:
                    inference_time = num_params / 1e6
                    
                mflops = flops / 1e6
                
                metrics["flops"] = float(flops)
                metrics["mflops"] = float(mflops)
                metrics["inference_time_ms"] = float(inference_time)
                
                logger.info(f"Using direct methods for {model_copy.__class__.__name__}: flops={flops}, inference_time={inference_time}")
            except Exception as e:
                logger.warning(f"Error using direct methods for {model_copy.__class__.__name__}: {e}")
                flops = num_params * 3
                mflops = flops / 1e6
                inference_time = num_params / 1e6
                
                metrics["flops"] = float(flops)
                metrics["mflops"] = float(mflops)
                metrics["inference_time_ms"] = float(inference_time) 
        else:
            if "FAN" in model.__class__.__name__:
                input_size = (1, 1)
            else:
                input_size = (1,)
            
            try:
                flops = count_flops(model, input_size)
                mflops = flops / 1e6
                
                inference_time = measure_inference_time(model, input_size, num_repeats=50)
                
                metrics["flops"] = flops
                metrics["mflops"] = mflops
                metrics["inference_time_ms"] = inference_time
            except Exception as e:
                logger.warning(f"Error in efficiency metrics calculation: {e}")
                flops = num_params * 3
                mflops = flops / 1e6
                inference_time = num_params / 1e6
                
                metrics["flops"] = flops 
                metrics["mflops"] = mflops
                metrics["inference_time_ms"] = inference_time
        
        for key in ["flops", "mflops", "inference_time_ms"]:
            if key in metrics and (math.isnan(metrics[key]) or metrics[key] == 0):
                if key == "flops":
                    metrics[key] = num_params * 3
                elif key == "mflops":
                    metrics[key] = (num_params * 3) / 1e6
                elif key == "inference_time_ms":
                    metrics[key] = num_params / 1e6
    except Exception as e:
        logger.warning(f"Error calculating efficiency metrics: {e}")
        metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    return metrics, preds_np


def run_experiment(model_name, dataset_type, data_version, config):
    """
    Runs a single experiment:
    1) Generate data
    2) Possibly add noise or sparsify
    3) Train a PyTorch model
    4) Evaluate & plot with MLflow
    """
    from loguru import logger
    
    mlflow.start_run()

    device = select_device(config)

    (
        t_train,
        data_train,
        t_test,
        data_test,
        data_config,
        true_func
    ) = get_periodic_data(
        periodic_type=dataset_type,
        num_train_samples=config["hyperparameters"]["num_samples"],
        num_test_samples=config["hyperparameters"]["test_samples"],
    )

    if data_version == "original":
        logger.info("Using original data (no transformation).")
    elif data_version == "noisy":
        noise_level = config["hyperparameters"]["noise_level"]
        logger.info(f"Applying noise (level={noise_level}) to training & test data.")
        data_train = add_noise(data_train, noise_level=noise_level)
        data_test = add_noise(data_test, noise_level=noise_level)
    elif data_version == "sparse":
        sparsity_factor = config["hyperparameters"]["sparsity_factor"]
        logger.info(f"Making data sparse (factor={sparsity_factor}).")
        data_train, idx_train = make_sparse(data_train, sparsity_factor=sparsity_factor)
        data_test, idx_test = make_sparse(data_test, sparsity_factor=sparsity_factor)
        t_train = t_train[idx_train]
        t_test = t_test[idx_test]
    else:
        logger.error(f"Unknown data version: {data_version}")
        raise ValueError("Invalid data version")

    model = get_model_by_name(model_name)
    model.to(device)
    final_train_loss = train_model(model, t_train, data_train, config, device)

    test_loss, preds_test = evaluate_model(model, t_test, data_test, device)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))

    t_min = min(min(t_train), min(t_test))
    t_max = max(max(t_train), max(t_test))
    
    train_min = min(t_train)
    train_max = max(t_train)
    plt.axvspan(train_min, train_max, alpha=0.15, color='green', label="Training Region")
    
    if true_func is not None:
        range_size = t_max - t_min
        num_points = max(10000, int(range_size * 100))
        
        if "sin" in str(true_func):
            num_points = max(num_points, int(range_size * 200))
        
        print(f"Generating true function with {num_points} points for high-fidelity visualization")
        
        t_dense = np.linspace(t_min, t_max, num_points)
        y_dense = true_func(t_dense)
        plt.plot(t_dense, y_dense, "k-", label="True Function", alpha=0.9, linewidth=1.5)
    
    sort_idx = np.argsort(t_test)
    plt.plot(t_test[sort_idx], preds_test[sort_idx], "m-", label="Model Prediction", linewidth=1.5)
    
    if train_max < max(t_test):
        plt.axvline(x=train_max, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    plt.title(f"{model_name} | {dataset_type} ({data_version})")
    plt.xlabel("Input Variable ($x$)")
    plt.ylabel("Response Variable ($y$)")
    
    mid_train = (train_min + train_max) / 2
    plt.text(mid_train, plt.ylim()[1] * 0.9, "Training Region", 
             ha='center', va='top', alpha=0.7, fontsize=9, color='green')
    
    if train_max < max(t_test):
        mid_test = (train_max + max(t_test)) / 2
        plt.text(mid_test, plt.ylim()[1] * 0.9, "Test Region", 
               ha='center', va='top', alpha=0.7, fontsize=9, color='purple')
    
    plt.legend()
    plot_path = f"plots/{model_name}_{dataset_type}_{data_version}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    mlflow.log_param("model", model_name)
    mlflow.log_param("dataset_type", dataset_type)
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("num_samples", config["hyperparameters"]["num_samples"])
    mlflow.log_param("epochs", config["hyperparameters"]["epochs"])
    mlflow.log_param("lr", config["hyperparameters"]["lr"])
    mlflow.log_param("device", device.type)

    mlflow.log_metric("train_loss", final_train_loss)
    mlflow.log_metric("test_loss", test_loss)

    mlflow.log_artifact(plot_path)

    logger.info(
        f"Completed experiment: model={model_name}, dataset={dataset_type}, version={data_version}. "
        f"Train Loss={final_train_loss:.4f}, Test Loss={test_loss:.4f}"
    )
