
import mlflow
import mlflow.pytorch
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.signal_gen import get_periodic_data
from utils.data_utils import add_noise, make_sparse
from src.models import get_model_by_name



def select_device(config):
    """
    Determine which device to use based on config and hardware availability.
    If config['hyperparameters']['device'] is set to 'mps', 'cuda', or 'cpu',
    we try that. Otherwise, we pick the best available by default.
    """
    # Attempt to read device preference from config
    device_str = config["hyperparameters"].get("device", None)

    # 1) If user explicitly asked for 'mps' and it's available, use that
    if device_str == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon).")
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    # 2) If user explicitly asked for 'cuda' and it's available, use that
    elif device_str == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device (NVIDIA GPU).")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    # 3) If user explicitly asked for 'cpu', or device_str is unknown
    elif device_str == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU device as requested.")

    # 4) Otherwise, pick the best available automatically
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("No device specified; using MPS for Apple Silicon.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("No device specified; using CUDA.")
        else:
            device = torch.device("cpu")
            logger.info("No device specified; using CPU.")

    return device


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
    # Convert to PyTorch tensors
    x_tensor = torch.from_numpy(t_train).float().unsqueeze(-1).to(device)
    y_tensor = torch.from_numpy(data_train).float().unsqueeze(-1).to(device)

    # Simple MSE loss
    criterion = nn.MSELoss()

    # Basic optimizer (e.g. Adam)
    lr = config["hyperparameters"].get("lr", 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = config["hyperparameters"].get("epochs", 10)
    batch_size = min(64, len(x_tensor))  # Just an example
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x_batch, y_batch in dataloader:
            # x_batch, y_batch already on 'device' since DataLoader is pulling from GPU Tensors
            optimizer.zero_grad()
            preds = model(x_batch)  # Forward pass
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    return epoch_loss  # Return the final training loss



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

    # Move data to the same device
    x_tensor = torch.from_numpy(t_test).float().unsqueeze(-1).to(device)
    y_tensor = torch.from_numpy(data_test).float().unsqueeze(-1).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(x_tensor)
        mse_loss = criterion(preds, y_tensor).item()

    # Convert preds back to numpy (on CPU)
    preds_np = preds.squeeze(-1).cpu().numpy()

    # Calculate metrics
    mse = mean_squared_error(data_test, preds_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(data_test, preds_np)
    r2 = r2_score(data_test, preds_np)

    # Calculate MAPE, handling potential division by zero
    try:
        mape = mean_absolute_percentage_error(data_test, preds_np) * 100
    except:
        # Handle cases where actual values contain zeros
        mape = (
            np.mean(
                np.abs((data_test - preds_np) / np.maximum(np.abs(data_test), 1e-10))
            )
            * 100
        )

    # Create metrics dictionary
    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

    return metrics, preds_np


def run_experiment(model_name, dataset_type, data_version, config):
    """
    Runs a single experiment:
    1) Generate data
    2) Possibly add noise or sparsify
    3) Train a PyTorch model
    4) Evaluate & plot with MLflow
    """
    mlflow.start_run()

    # Determine which device to use
    device = select_device(config)

    # (1) Generate the base dataset + retrieve underlying function
    # We assume get_periodic_data returns (t_train, data_train, t_test, data_test, data_config, true_func)
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

    # (2) Apply data transformation if needed
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
        # Also shrink time arrays
        t_train = t_train[idx_train]
        t_test = t_test[idx_test]
    else:
        logger.error(f"Unknown data version: {data_version}")
        raise ValueError("Invalid data version")

    # (3) Retrieve & train the model
    model = get_model_by_name(model_name)
    model.to(device)  # Move model to device
    final_train_loss = train_model(model, t_train, data_train, config, device)

    # (4) Evaluate on test set
    test_loss, preds_test = evaluate_model(model, t_test, data_test, device)

    # (5) Plot everything
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))

    # Underlying "true" function (red dashed)
    if true_func is not None:
        t_dense = np.linspace(min(t_test), max(t_test), 2000)
        y_dense = true_func(t_dense)
        plt.plot(t_dense, y_dense, "r--", label="True Function", alpha=0.7)

    # Training data (blue dots)
    plt.scatter(t_train, data_train, color="blue", s=10, alpha=0.5, label="Train Data")

    # Test data (green dots)
    plt.scatter(t_test, data_test, color="green", s=10, alpha=0.5, label="Test Data")

    # Model predictions (magenta line)
    sort_idx = np.argsort(t_test)
    plt.plot(t_test[sort_idx], preds_test[sort_idx], "m-", label="Model Prediction")

    plt.title(f"{model_name} | {dataset_type} ({data_version})")
    plt.legend()
    plot_path = f"plots/{model_name}_{dataset_type}_{data_version}.png"
    plt.savefig(plot_path)
    plt.close()

    # (6) Log metrics & artifacts with MLflow

    mlflow.log_param("model", model_name)
    mlflow.log_param("dataset_type", dataset_type)
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("num_samples", config["hyperparameters"]["num_samples"])
    mlflow.log_param("epochs", config["hyperparameters"]["epochs"])
    mlflow.log_param("lr", config["hyperparameters"]["lr"])
    mlflow.log_param("device", device.type)

    mlflow.log_metric("train_loss", final_train_loss)
    mlflow.log_metric("test_loss", test_loss)

    # Log the plot
    mlflow.log_artifact(plot_path)

        # (Optional) Save the trained model to MLflow
        # mlflow.pytorch.log_model(model, artifact_path="model")

    logger.info(
        f"Completed experiment: model={model_name}, dataset={dataset_type}, version={data_version}. "
        f"Train Loss={final_train_loss:.4f}, Test Loss={test_loss:.4f}"
    )



