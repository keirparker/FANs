#!/usr/bin/env python
"""
Time series data loading and preprocessing module.

This module provides utilities to load and preprocess time series datasets
for forecasting tasks, making them compatible with the existing pipeline.

Author: GitHub Copilot for keirparker
Last updated: 2025-03-02
"""

import pandas as pd
import numpy as np
import os
from loguru import logger
from typing import Tuple, Dict, List, Any



def load_ts_dataset(
    config: Dict[str, Any], dataset_name: str = "etth1"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], None]:
    """
    Load and preprocess a time series dataset based on configuration.
    Returns data in a format compatible with the main pipeline.

    Args:
        config: Configuration dictionary containing hyperparameters
        dataset_name: Name of the dataset to load

    Returns:
        t_train: Time values for training (converted to numeric for PyTorch compatibility)
        data_train: Training data
        t_test: Time values for testing (converted to numeric for PyTorch compatibility)
        data_test: Test data
        data_config: Configuration dictionary with dataset metadata
        true_func: Always None for time series data (no ground truth function)
    """
    # Extract parameters from config
    hp = config["hyperparameters"]
    seq_len = hp.get("seq_len", 96)
    pred_len = hp.get("pred_len", 24)
    target_col = hp.get("target_col", "OT")
    normalize = hp.get("normalize", True)
    train_ratio = hp.get("train_ratio", 0.7)
    val_ratio = hp.get("val_ratio", 0.1)

    # Dataset-specific file paths
    data_dir = hp.get("data_dir", "data")
    dataset_paths = {
        "etth1": os.path.join(data_dir, 'raw','ETT', "ETTh1.csv"),
        "etth2": os.path.join(data_dir, 'raw','ETT', "ETTh2.csv"),
        "ettm1": os.path.join(data_dir, 'raw','ETT', "ETTm1.csv")
    }

    file_path = dataset_paths.get(dataset_name.lower())
    if not file_path:
        raise ValueError(f"Unknown time series dataset: {dataset_name}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    # Load the data
    logger.info(f"Loading {dataset_name} dataset from {file_path}")
    df = pd.read_csv(file_path)

    # Determine columns
    timestamp_col = hp.get("timestamp_col", "date")
    feature_cols = hp.get("feature_cols", None)

    # Parse timestamps
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Select feature columns if not specified
    if feature_cols is None:
        feature_cols = [col for col in df.select_dtypes(include=np.number).columns]

    # Ensure target column is included
    if target_col not in feature_cols and target_col in df.columns:
        feature_cols.append(target_col)

    # Split into train/val/test based on time
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Create data info dictionary
    data_info = {
        "total_samples": n,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "timestamp_col": timestamp_col,
        "input_dim": len(feature_cols),
    }

    # Normalize if requested
    if normalize:
        # Calculate stats from training data
        means = train_df[feature_cols].mean()
        stds = train_df[feature_cols].std()
        stds.replace(0, 1, inplace=True)  # Avoid division by zero

        # Store normalization parameters
        data_info["means"] = means.to_dict()
        data_info["stds"] = stds.to_dict()

        # Apply normalization
        for col in feature_cols:
            train_df[col] = (train_df[col] - means[col]) / stds[col]
            val_df[col] = (val_df[col] - means[col]) / stds[col]
            test_df[col] = (test_df[col] - means[col]) / stds[col]

    # Create sequences
    t_train, x_train, y_train = _create_forecast_sequences(
        train_df, seq_len, pred_len, feature_cols, target_col, timestamp_col
    )

    # Add validation data to training if specified
    if hp.get("include_val_in_train", True):
        t_val, x_val, y_val = _create_forecast_sequences(
            val_df, seq_len, pred_len, feature_cols, target_col, timestamp_col
        )

        if len(t_val) > 0:
            t_train = np.vstack((t_train, t_val))
            x_train = np.vstack((x_train, x_val))
            y_train = np.vstack((y_train, y_val))

    # Create test sequences
    t_test, x_test, y_test = _create_forecast_sequences(
        test_df, seq_len, pred_len, feature_cols, target_col, timestamp_col
    )

    logger.info(f"Created {len(t_train)} training and {len(t_test)} test sequences")

    # Convert datetime timestamps to numeric values (days since Unix epoch)
    # This makes them compatible with PyTorch tensors
    def convert_datetime_to_numeric(dt_array):
        # Get the first timestamp for each sequence
        first_timestamps = np.array([ts[0] for ts in dt_array])
        # Convert to days since Unix epoch for numerical stability
        numeric_timestamps = np.array(
            [
                (
                    pd.Timestamp(ts).to_datetime64().astype("datetime64[D]")
                    - np.datetime64("1970-01-01")
                ).astype(float)
                for ts in first_timestamps
            ]
        )
        return numeric_timestamps

    # Convert timestamps to numeric
    t_train_numeric = convert_datetime_to_numeric(t_train)
    t_test_numeric = convert_datetime_to_numeric(t_test)

    # Store original timestamps in data_config for reference
    data_info["original_timestamps"] = {"train": t_train, "test": t_test}

    # Flatten inputs and concatenate with targets
    data_train = np.hstack((x_train.reshape(x_train.shape[0], -1), y_train))
    data_test = np.hstack((x_test.reshape(x_test.shape[0], -1), y_test))

    # Create config with all relevant information
    data_config = {
        **hp,  # Include all hyperparameters
        "input_dim": x_train.shape[2],  # Number of features
        "seq_len": seq_len,
        "pred_len": pred_len,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "dataset_name": dataset_name,
        **data_info,  # Include dataset info
    }

    return t_train_numeric, data_train, t_test_numeric, data_test, data_config, None


def _create_forecast_sequences(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    feature_cols: List[str],
    target_col: str,
    timestamp_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create input-output sequences for time series forecasting.

    Args:
        df: DataFrame containing time series data
        seq_len: Length of input sequence (lookback window)
        pred_len: Length of prediction horizon
        feature_cols: Columns to use as features
        target_col: Column to predict
        timestamp_col: Column with timestamps

    Returns:
        times: Array of timestamps for each sequence
        inputs: Array of input sequences [n_samples, seq_len, n_features]
        targets: Array of target sequences [n_samples, pred_len]
    """
    n_samples = len(df) - seq_len - pred_len + 1
    n_features = len(feature_cols)

    if n_samples <= 0:
        raise ValueError(
            f"DataFrame too short for sequence length {seq_len} and prediction length {pred_len}"
        )

    # Pre-allocate arrays
    inputs = np.zeros((n_samples, seq_len, n_features))
    targets = np.zeros((n_samples, pred_len))
    times = np.empty((n_samples, pred_len), dtype="datetime64[ns]")

    feature_data = df[feature_cols].values
    target_data = df[[target_col]].values.flatten()
    time_data = df[timestamp_col].values

    # Create sequences
    for i in range(n_samples):
        # Input sequence is all features
        inputs[i] = feature_data[i : i + seq_len]

        # Target is just the target column
        targets[i] = target_data[i + seq_len : i + seq_len + pred_len]

        # Store timestamps for the prediction period
        times[i] = time_data[i + seq_len : i + seq_len + pred_len]

    return times, inputs, targets


def get_etth1_data(
    seq_len: int = None,
    pred_len: int = None,
    target_col: str = None,
    config: Dict[str, Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], None]:
    """
    Get ETTh1 data in a format compatible with the main pipeline.
    This is a convenience wrapper around load_ts_dataset for backward compatibility.

    Args:
        seq_len: Input sequence length (if None, uses config)
        pred_len: Prediction horizon length (if None, uses config)
        target_col: Target column to predict (if None, uses config)
        config: Configuration dictionary (if None, uses default values)

    Returns:
        Same as load_ts_dataset
    """
    # Create default config if not provided
    if config is None:
        config = {
            "hyperparameters": {
                "seq_len": seq_len or 96,
                "pred_len": pred_len or 24,
                "target_col": target_col or "OT",
                "normalize": True,
            }
        }
    else:
        # Override config with explicit parameters if provided
        if seq_len is not None:
            config["hyperparameters"]["seq_len"] = seq_len
        if pred_len is not None:
            config["hyperparameters"]["pred_len"] = pred_len
        if target_col is not None:
            config["hyperparameters"]["target_col"] = target_col

    return load_ts_dataset(config, dataset_name="etth1")
