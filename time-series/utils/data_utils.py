#!/usr/bin/env python
"""
Advanced data processing utilities for high-performance ML experimentation.

This module provides optimized, GPU-accelerated data processing functions
with extensive error handling, performance optimizations, and modern ML practices.

Author: GitHub Copilot for keirparker
Last updated: 2025-02-26
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import os
from typing import Tuple, Optional, Union, Callable, List, Dict, Any
from loguru import logger
from contextlib import contextmanager
import time
from functools import wraps
import psutil


def performance_metrics(func):
    """
    Decorator to measure and log performance metrics of data operations.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

        result = func(*args, **kwargs)

        elapsed_time = time.perf_counter() - start_time
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        mem_diff = mem_after - mem_before

        logger.debug(f"{func.__name__} completed in {elapsed_time:.4f}s | "
                     f"Memory change: {mem_diff:.2f}MB")
        return result
    return wrapper


@contextmanager
def gpu_transfer_context(tensor_list: List[torch.Tensor], device: torch.device):
    """
    Context manager for efficient batch transfer to GPU with memory optimization.
    """
    transferred = []
    try:
        for tensor in tensor_list:
            if tensor is not None:
                transferred.append(tensor.to(device, non_blocking=True))
            else:
                transferred.append(None)
        yield transferred
    finally:
        for i in range(len(transferred)):
            transferred[i] = None


@performance_metrics
def add_noise(
        data: np.ndarray,
        noise_level: float = 0.1,
        noise_type: str = "gaussian",
        seed: Optional[int] = None,
        preserve_range: bool = True
) -> np.ndarray:
    """
    Add sophisticated noise patterns to data with distribution control.

    Args:
        data: Input data array of shape (N,) or (N,C)
        noise_level: Noise magnitude (stdev for gaussian, range for uniform)
        noise_type: Noise distribution type: "gaussian", "uniform", "laplace",
                   "salt_pepper", "poisson", or "speckle"
        seed: Random seed for reproducibility
        preserve_range: Ensure noisy data maintains similar min/max range as original

    Returns:
        np.ndarray: Data with carefully controlled noise

    Raises:
        ValueError: If invalid noise_type specified
    """
    if seed is not None:
        np.random.seed(seed)

    if preserve_range:
        orig_min, orig_max = np.min(data), np.max(data)
        orig_range = orig_max - orig_min

    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level, size=data.shape)
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_level, noise_level, size=data.shape)
    elif noise_type == "laplace":
        noise = np.random.laplace(0, noise_level, size=data.shape)
    elif noise_type == "salt_pepper":
        noise = np.zeros_like(data)
        salt_mask = np.random.random(size=data.shape) < (noise_level / 2)
        noise[salt_mask] = 1.0
        pepper_mask = np.random.random(size=data.shape) < (noise_level / 2)
        noise[pepper_mask] = -1.0
    elif noise_type == "poisson":
        data_range = np.max(data) - np.min(data)
        if data_range > 0:
            scaled_data = (data - np.min(data)) / data_range * 255
            noisy_data = np.random.poisson(scaled_data) / 255.0 * data_range + np.min(data)
            noise = noisy_data - data
        else:
            noise = np.zeros_like(data)
    elif noise_type == "speckle":
        noise = data * np.random.normal(0, noise_level, size=data.shape)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    noisy_data = data + noise

    if preserve_range:
        noisy_min, noisy_max = np.min(noisy_data), np.max(noisy_data)
        noisy_range = noisy_max - noisy_min

        if noisy_range > 0:
            noisy_data = (noisy_data - noisy_min) / noisy_range * orig_range + orig_min

    return noisy_data


@performance_metrics
def make_sparse(
        data: np.ndarray,
        t: Optional[np.ndarray] = None,
        sparsity_factor: Union[int, float] = 5,
        method: str = "uniform",
        keep_endpoints: bool = True,
        importance_sampling: bool = False,
        importance_fn: Optional[Callable] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sparse data with advanced subsampling techniques.

    Args:
        data: Data array of shape (N,) or (N,C)
        t: Corresponding time/index points (optional)
        sparsity_factor: If int: Keep only every nth point
                         If float < 1.0: Keep this fraction of points
        method: Sampling method - "uniform", "random", "stratified",
                "gradient_based", "kmeans"
        keep_endpoints: Always keep first and last points
        importance_sampling: Sample based on data importance (variation)
        importance_fn: Function to calculate point importance (default: gradient)

    Returns:
        tuple: (sparse_data, sparse_indices)

    Raises:
        ValueError: If invalid method or parameters are specified
    """
    n = len(data)
    if n <= 2:
        return data, np.arange(n)

    if isinstance(sparsity_factor, float) and sparsity_factor < 1.0:
        k = max(2, int(n * sparsity_factor))  # Number of points to keep
    else:
        k = max(2, n // int(sparsity_factor))

    k = min(k, n)

    if t is None:
        t = np.arange(n)

    if method == "uniform":
        indices = np.linspace(0, n-1, k, dtype=int)

    elif method == "random":
        local_rng = np.random.RandomState(seed) if seed is not None else np.random
        indices = np.sort(local_rng.choice(n, size=k, replace=False))

    elif method == "stratified":
        local_rng = np.random.RandomState(seed) if seed is not None else np.random
        bin_edges = np.linspace(0, n, k+1, dtype=int)
        indices = np.array([
            local_rng.choice(np.arange(bin_edges[i], bin_edges[i+1]))
            for i in range(k)
        ])

    elif method == "gradient_based":
        if len(data.shape) > 1:
            gradients = np.sum(np.abs(np.diff(data, axis=0)), axis=1)
        else:
            gradients = np.abs(np.diff(data))

        padded_grads = np.pad(gradients, (1, 0), mode='constant')

        top_indices = np.argsort(padded_grads)[-k:]
        indices = np.sort(top_indices)

    elif method == "kmeans":
        try:
            from sklearn.cluster import KMeans
            coords = np.column_stack((t, data))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)

            indices = np.array([
                np.where(labels == i)[0][
                    np.argmin(np.linalg.norm(
                        coords[labels == i] - kmeans.cluster_centers_[i], axis=1
                    ))
                ]
                for i in range(k)
            ])
            indices = np.sort(indices)
        except ImportError:
            logger.warning("sklearn not available, falling back to uniform sampling")
            indices = np.linspace(0, n-1, k, dtype=int)

    elif importance_sampling:
        if importance_fn is None:
            if len(data.shape) > 1:
                importance = np.sum(np.abs(np.diff(data, axis=0)), axis=1)
                importance = np.pad(importance, (1, 0), mode='constant')
            else:
                g1 = np.abs(np.diff(data))
                g1 = np.pad(g1, (1, 0), mode='constant')
                g2 = np.abs(np.diff(data, n=2))
                g2 = np.pad(g2, (1, 1), mode='constant')
                importance = g1 + 0.5 * g2
        else:
            importance = importance_fn(data)

        if np.sum(importance) > 0:
            probs = importance / np.sum(importance)
            indices = np.sort(np.random.choice(n, size=k, replace=False, p=probs))
        else:
            indices = np.sort(np.random.choice(n, size=k, replace=False))

    else:
        raise ValueError(f"Unknown sampling method: {method}")

    if keep_endpoints:
        if 0 not in indices:
            indices = np.sort(np.append(indices, 0))
        if n-1 not in indices:
            indices = np.sort(np.append(indices, n-1))

    sparse_data = data[indices]

    return sparse_data, indices


@performance_metrics
def split_data(
        t: np.ndarray,
        data: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: Optional[int] = None,
        stratify: Optional[np.ndarray] = None,
        temporal: bool = True,
        gap_size: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Advanced data splitting with multiple strategies for time series.

    Args:
        t: Time points array
        data: Data array
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        stratify: Optional array for stratified sampling
        temporal: If True, preserves temporal order in splits
        gap_size: Fraction of data to use as gaps between splits (avoiding leakage)

    Returns:
        tuple: (t_train, data_train, t_val, data_val, t_test, data_test)
    """
    n = len(t)
    if n <= 3:
        return t, data, t, data, t, data

    if random_state is not None:
        np.random.seed(random_state)

    total_gap = gap_size * 2  # Two gaps: train-val and val-test
    available_fraction = 1.0 - total_gap

    if available_fraction <= 0:
        logger.warning("Gap size too large, reducing to fit data")
        gap_size = 0
        total_gap = 0
        available_fraction = 1.0

    effective_test_size = test_size / available_fraction
    effective_val_size = val_size / available_fraction

    if temporal:
        train_end = int(n * (1 - effective_test_size - effective_val_size - gap_size * 2))
        val_start = train_end + max(1, int(n * gap_size))
        val_end = val_start + int(n * effective_val_size)
        test_start = val_end + max(1, int(n * gap_size))

        train_indices = np.arange(0, train_end)
        val_indices = np.arange(val_start, val_end)
        test_indices = np.arange(test_start, n)

    else:
        if stratify is not None:
            try:
                from sklearn.model_selection import train_test_split

                combined_size = effective_test_size + effective_val_size
                train_indices, temp_indices = train_test_split(
                    np.arange(n),
                    test_size=combined_size,
                    random_state=random_state,
                    stratify=stratify
                )

                relative_val_size = effective_val_size / combined_size
                val_indices, test_indices = train_test_split(
                    temp_indices,
                    test_size=(1 - relative_val_size),
                    random_state=random_state if random_state is None else random_state+1,
                    stratify=stratify[temp_indices] if stratify is not None else None
                )

            except ImportError:
                logger.warning("sklearn not available, falling back to random split")
                indices = np.random.permutation(n)
                train_end = int(n * (1 - effective_test_size - effective_val_size))
                val_end = train_end + int(n * effective_val_size)
                train_indices = indices[:train_end]
                val_indices = indices[train_end:val_end]
                test_indices = indices[val_end:]
        else:
            indices = np.random.permutation(n)
            train_end = int(n * (1 - effective_test_size - effective_val_size))
            val_end = train_end + int(n * effective_val_size)
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

    t_train, data_train = t[train_indices], data[train_indices]
    t_val, data_val = t[val_indices], data[val_indices]
    t_test, data_test = t[test_indices], data[test_indices]

    logger.info(f"Data split: {len(t_train)} train ({len(t_train)/n:.1%}), "
                f"{len(t_val)} validation ({len(t_val)/n:.1%}), "
                f"{len(t_test)} test ({len(t_test)/n:.1%})")

    return t_train, data_train, t_val, data_val, t_test, data_test


class AdaptiveTimeSeriesDataset(Dataset):
    """
    Advanced PyTorch Dataset for time series with dynamic transformations,
    caching, and multi-resolution support.
    """
    def __init__(
        self,
        t: np.ndarray,
        data: np.ndarray,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        cache_size: int = 1000,
        precision: torch.dtype = torch.float32,
        return_indices: bool = False,
        augmentations: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the advanced dataset.

        Args:
            t: Time points array
            data: Data array
            transform: Transform to apply to inputs
            target_transform: Transform to apply to targets
            device: Device to store tensors on (None = CPU)
            cache_size: Size of transformation cache for performance
            precision: Floating point precision for tensors
            return_indices: Whether to return indices with samples
            augmentations: List of augmentation configs to apply dynamically
            seed: Random seed for reproducible augmentations and transformations
        """
        self.return_indices = return_indices
        self.cache_size = cache_size
        self.transform = transform
        self.target_transform = target_transform
        self.augmentations = augmentations or []
        self.seed = seed

        self.x = torch.tensor(t, dtype=precision)
        self.y = torch.tensor(data, dtype=precision)

        if len(self.x.shape) == 1:
            self.x = self.x.unsqueeze(-1)
        if len(self.y.shape) == 1:
            self.y = self.y.unsqueeze(-1)

        if device is not None and device.type != 'cuda':
            self.x = self.x.to(device)
            self.y = self.y.to(device)

        self.transform_cache = {}

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_single_item(idx)
        else:
            return self._get_batch_items(idx)

    def _get_single_item(self, idx: int):
        """Handle fetching and processing a single item"""
        if self.cache_size > 0 and idx in self.transform_cache:
            result = self.transform_cache[idx]
        else:
            x, y = self.x[idx], self.y[idx]

            if self.transform is not None:
                x = self.transform(x)

            if self.target_transform is not None:
                y = self.target_transform(y)

            if self.augmentations:
                if hasattr(self, 'seed') and self.seed is not None:
                    sample_seed = self.seed + idx
                    rng = np.random.RandomState(sample_seed)
                    
                    if rng.random() < 0.5:  # 50% chance to augment
                        aug_idx = rng.randint(0, len(self.augmentations))
                        aug = self.augmentations[aug_idx]
                        x, y = self._apply_augmentation(x, y, aug, rng)
                else:
                    if np.random.random() < 0.5:  # 50% chance to augment
                        aug = np.random.choice(self.augmentations)
                        x, y = self._apply_augmentation(x, y, aug)

            result = ((idx, x, y) if self.return_indices else (x, y))

            if self.cache_size > 0:
                if len(self.transform_cache) >= self.cache_size:
                    remove_key = next(iter(self.transform_cache))
                    del self.transform_cache[remove_key]
                self.transform_cache[idx] = result

        return result

    def _get_batch_items(self, indices):
        """Handle fetching and processing multiple items"""
        items = [self._get_single_item(idx) for idx in indices]

        if self.return_indices:
            indices, x_tensors, y_tensors = zip(*items)
            return torch.tensor(indices), torch.stack(x_tensors), torch.stack(y_tensors)
        else:
            x_tensors, y_tensors = zip(*items)
            return torch.stack(x_tensors), torch.stack(y_tensors)

    def _apply_augmentation(self, x, y, aug_config, rng=None):
        """
        Apply an augmentation to a sample
        
        Args:
            x: Input tensor
            y: Target tensor
            aug_config: Augmentation configuration dictionary
            rng: Optional numpy RandomState for reproducible augmentation
        """
        aug_type = aug_config.get('type')
        
        if rng is None:
            generator = torch.Generator()
            if hasattr(self, 'seed') and self.seed is not None:
                generator.manual_seed(self.seed)
        else:
            generator = None  # use numpy random values directly

        if aug_type == 'noise':
            noise_level = aug_config.get('level', 0.05)
            if rng is not None:
                noise_np = rng.randn(*x.shape) * noise_level
                noise = torch.tensor(noise_np, dtype=x.dtype, device=x.device)
            else:
                noise = torch.randn_like(x, generator=generator) * noise_level
            return x + noise, y

        elif aug_type == 'scale':
            std = aug_config.get('std', 0.1)
            if rng is not None:
                scale_factor = aug_config.get('factor', 1.0) + rng.randn() * std
            else:
                scale_factor = aug_config.get('factor', 1.0) + torch.randn(1, generator=generator).item() * std
            return x * scale_factor, y * scale_factor

        elif aug_type == 'shift':
            std = aug_config.get('std', 0.1)
            if rng is not None:
                shift_amount = aug_config.get('amount', 0.0) + rng.randn() * std
            else:
                shift_amount = aug_config.get('amount', 0.0) + torch.randn(1, generator=generator).item() * std
            return x + shift_amount, y

        elif aug_type == 'flip':
            flip_prob = aug_config.get('prob', 0.5)
            if rng is not None:
                do_flip = rng.random() < flip_prob
            else:
                do_flip = torch.rand(1, generator=generator).item() < flip_prob
                
            if do_flip:
                return -x, -y
            return x, y

        return x, y

    def to(self, device):
        """Move dataset to specified device"""
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.transform_cache = {}
        return self

    def statistics(self):
        """Return dataset statistics"""
        x_mean, x_std = self.x.mean().item(), self.x.std().item()
        y_mean, y_std = self.y.mean().item(), self.y.std().item()
        return {
            'x_stats': {'mean': x_mean, 'std': x_std, 'min': self.x.min().item(), 'max': self.x.max().item()},
            'y_stats': {'mean': y_mean, 'std': y_std, 'min': self.y.min().item(), 'max': self.y.max().item()}
        }


@performance_metrics
def prepare_data_loaders(
    t_train: np.ndarray,
    data_train: np.ndarray,
    config: Dict[str, Any],
    t_val: Optional[np.ndarray] = None,
    data_val: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    sampler_type: str = "random"
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create optimized DataLoaders with advanced batching strategies.

    Args:
        t_train: Training time points
        data_train: Training data values
        config: Configuration dictionary
        t_val: Validation time points (optional)
        data_val: Validation data values (optional)
        device: PyTorch device (optional)
        sampler_type: Type of sampling strategy ("random", "weighted", "sequential", "balanced")

    Returns:
        tuple: (train_loader, val_loader)
    """
    batch_size = config["hyperparameters"].get("batch_size", 256)  # Larger batch size for V100
    
    if os.environ.get("AWS_EXECUTION_ENV", "").startswith("AWS_ECS"):
        default_workers = 4
    else:
        default_workers = min(8, (os.cpu_count() or 4) // 2)
        
    num_workers = config["hyperparameters"].get("num_workers", default_workers)
    
    pin_memory = config["hyperparameters"].get("pin_memory", False)
    if device is not None and device.type == "cuda":
        pin_memory = False
    
    persistent_workers = config["hyperparameters"].get("persistent_workers", num_workers > 0)
    precision = config["hyperparameters"].get("precision", "float32")
    prefetch_factor = config["hyperparameters"].get("prefetch_factor", 2)
    drop_last = config["hyperparameters"].get("drop_last", False)

    torch_precision = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }.get(precision, torch.float32)

    batch_size = min(batch_size, len(t_train))

    seed = config.get("random_seed", 42)
    
    augmentations = [
        {'type': 'noise', 'level': 0.03},
        {'type': 'scale', 'factor': 1.0, 'std': 0.1},
        {'type': 'shift', 'amount': 0.0, 'std': 0.1}
    ] if config["hyperparameters"].get("use_augmentation", False) else None

    train_dataset = AdaptiveTimeSeriesDataset(
        t_train, data_train,
        precision=torch_precision,
        device=device,
        augmentations=augmentations,
        seed=seed  # pass seed for reproducible augmentations
    )

    if sampler_type == "weighted":
        gradients = np.abs(np.diff(data_train, prepend=data_train[0]))
        weights = torch.tensor(gradients, dtype=torch.float32)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
    elif sampler_type == "balanced":
        n_bins = 10
        data_range = np.ptp(data_train)
        bin_width = data_range / n_bins if data_range > 0 else 1.0
        bin_indices = np.minimum(n_bins-1, np.floor((data_train - np.min(data_train)) / bin_width)).astype(int)

        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_weights = 1.0 / (bin_counts + 1)

        weights = torch.tensor([bin_weights[b] for b in bin_indices], dtype=torch.float32)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
    elif sampler_type == "sequential":
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last
    )

    val_loader = None
    if t_val is not None and data_val is not None:
        val_dataset = AdaptiveTimeSeriesDataset(
            t_val, data_val,
            precision=torch_precision,
            device=device,
            augmentations=None,
            seed=seed
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=persistent_workers
        )

    train_stats = train_dataset.statistics()
    logger.debug(f"Train dataset stats: x_mean={train_stats['x_stats']['mean']:.3f}, "
                f"y_mean={train_stats['y_stats']['mean']:.3f}, "
                f"y_std={train_stats['y_stats']['std']:.3f}")

    return train_loader, val_loader
