"""Utilities for measuring model convergence."""

import numpy as np
import time
from loguru import logger
import ray

def init_ray_for_mac(num_cpus=None):
    """Initialize Ray with settings optimized for M2 Macbook.
    
    Args:
        num_cpus: Number of CPUs to use. If None, uses all available minus 2.
        
    Returns:
        bool: True if Ray was successfully initialized, False otherwise
    """
    if ray.is_initialized():
        return True

    import multiprocessing
    
    if num_cpus is None:
        # Leave 2 CPUs free for system operations on M2 Mac
        available_cpus = multiprocessing.cpu_count()
        num_cpus = max(1, available_cpus - 2)
    
    try:
        ray.init(
            num_cpus=num_cpus,
            include_dashboard=False,
            ignore_reinit_error=True,
            _temp_dir="/tmp/ray_temp",  # Prevent permissions issues on macOS
            _system_config={
                # MacOS-specific configurations
                "worker_register_timeout_seconds": 60,
                "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}}',
                "max_io_workers": 4,  # Reduce I/O worker threads for Mac
                "object_store_full_delay_ms": 100  # More aggressive memory management
            },
            logging_level=logging.WARNING
        )
        logger.info(f"Ray initialized with {num_cpus} CPUs")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize Ray: {e}. Will continue without parallel processing.")
        return False


def calculate_convergence_speed(history, target_metric='val_loss', threshold=0.1):
    """Calculate how quickly a model converges to within a threshold of its best performance."""
    if target_metric not in history or not history[target_metric]:
        logger.warning(f"Target metric '{target_metric}' not found in history")
        return None

    values = history[target_metric]
    
    is_loss = target_metric.endswith('loss')
    
    best_value = min(values) if is_loss else max(values)
    best_epoch = values.index(best_value) + 1  # +1 because epochs are 1-indexed
    
    if is_loss:
        threshold_value = best_value * (1 + threshold)
    else:
        threshold_value = best_value * (1 - threshold)
    
    convergence_epoch = None
    for i, value in enumerate(values):
        if (is_loss and value <= threshold_value) or (not is_loss and value >= threshold_value):
            convergence_epoch = i + 1  # +1 because epochs are 1-indexed
            break
            
    return convergence_epoch


def calculate_training_efficiency(history, target_metric='val_loss', threshold=0.1):
    """
    Calculate several efficiency metrics for a model's training process.
    
    Args:
        history: Dictionary containing training history with metrics by epoch
        target_metric: The metric to use for measuring convergence (default: 'val_loss')
        threshold: Relative threshold for determining convergence as a fraction (default: 0.1 = 10%)
        
    Returns:
        dict: Dictionary containing efficiency metrics
    """
    convergence_epoch = calculate_convergence_speed(history, target_metric, threshold)
    
    if convergence_epoch is None:
        logger.warning("Model did not converge within the specified threshold")
        return {
            'converged': False,
            'convergence_epoch': None,
            'convergence_speed': None,
            'final_value': None,
            'best_value': None,
            'avg_epoch_time': None,
            'total_convergence_time': None
        }
    
    values = history[target_metric]

    is_loss = target_metric.endswith('loss')
    
    best_value = min(values) if is_loss else max(values)
    final_value = values[-1]
    
    avg_epoch_time = None
    total_convergence_time = None
    
    if 'training_time' in history and len(history['training_time']) > 0:
        avg_epoch_time = np.mean(history['training_time'][:convergence_epoch])
        total_convergence_time = sum(history['training_time'][:convergence_epoch])
    
    return {
        'converged': True,
        'convergence_epoch': convergence_epoch,
        'convergence_speed': 1.0 / convergence_epoch if convergence_epoch > 0 else 0,
        'final_value': final_value,
        'best_value': best_value,
        'avg_epoch_time': avg_epoch_time,
        'total_convergence_time': total_convergence_time
    }


@ray.remote
def calculate_model_efficiency_ray(model_name, history, target_metric='val_loss', threshold=0.1):
    """Calculate efficiency metrics for a single model (Ray remote function)"""
    efficiency = calculate_training_efficiency(history, target_metric, threshold)
    return model_name, efficiency


def compare_convergence(histories, target_metric='val_loss', threshold=0.1, use_ray=True, num_cpus=None):
    """
    Compare convergence speed between multiple models using Ray for parallelization.
    
    Args:
        histories: Dictionary of model names to their training histories
        target_metric: The metric to use for measuring convergence (default: 'val_loss')
        threshold: Relative threshold for determining convergence as a fraction (default: 0.1 = 10%)
        use_ray: Whether to use Ray for parallel processing (default: True)
        num_cpus: Number of CPUs to use for Ray. If None, uses all available minus 2.
        
    Returns:
        dict: Dictionary containing convergence comparison results
    """
    # If only one model or Ray is disabled, use sequential processing
    if len(histories) <= 1 or not use_ray:
        results = {}
        for model_name, history in histories.items():
            efficiency = calculate_training_efficiency(history, target_metric, threshold)
            results[model_name] = efficiency
    else:
        # Try to use Ray for parallel processing
        try:
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                init_ray_for_mac(num_cpus)
                
            # Submit tasks to Ray
            tasks = [calculate_model_efficiency_ray.remote(model_name, history, target_metric, threshold) 
                    for model_name, history in histories.items()]
            
            # Get results
            model_results = ray.get(tasks)
            results = {model_name: efficiency for model_name, efficiency in model_results}
            
        except Exception as e:
            logger.warning(f"Failed to use Ray for parallel processing: {e}. Falling back to sequential processing.")
            # Fallback to sequential processing
            results = {}
            for model_name, history in histories.items():
                efficiency = calculate_training_efficiency(history, target_metric, threshold)
                results[model_name] = efficiency
    
    # Process the results (same for both parallel and sequential)
    converged_models = {name: data for name, data in results.items()
                      if data['converged']}
    
    if converged_models:
        sorted_models = sorted(converged_models.items(),
                              key=lambda x: x[1]['convergence_speed'], 
                              reverse=True)
        
        fastest_model = sorted_models[0][0]
        fastest_epoch = converged_models[fastest_model]['convergence_epoch']
        
        for model in results:
            if results[model]['converged']:
                model_epoch = results[model]['convergence_epoch']
                results[model]['relative_speed'] = fastest_epoch / model_epoch
            else:
                results[model]['relative_speed'] = 0
    
    return results


def detect_convergence_plateau(history, target_metric='val_loss', window_size=5, min_change=0.001):
    """
    Detect when a model reaches a plateau in its learning.
    
    Args:
        history: Dictionary containing training history with metrics by epoch
        target_metric: The metric to use for measuring convergence (default: 'val_loss')
        window_size: Number of epochs to consider for plateau detection (default: 5)
        min_change: Minimum relative change to consider as progress (default: 0.001 = 0.1%)
        
    Returns:
        int: Epoch at which the model plateaued, or None if no plateau detected
    """
    if target_metric not in history or len(history[target_metric]) < window_size:
        return None
        
    values = history[target_metric]
    
    is_loss = target_metric.endswith('loss')
    
    for i in range(len(values) - window_size):
        window_start = values[i]
        window_end = values[i + window_size - 1]
        
        if abs(window_start) < 1e-10:  # avoid division by zero
            rel_improvement = abs(window_end - window_start)
        else:
            rel_improvement = abs(window_end - window_start) / abs(window_start)
        
        if rel_improvement < min_change:
            if is_loss and window_end > window_start:
                continue
            if not is_loss and window_end < window_start:
                continue
                
            return i + window_size  #  plateau point
    
    return None


@ray.remote
def detect_plateau_ray(model_name, history, target_metric='val_loss', window_size=5, min_change=0.001):
    """Ray remote function to detect plateau for a single model"""
    plateau_epoch = detect_convergence_plateau(history, target_metric, window_size, min_change)
    return model_name, plateau_epoch


def detect_plateaus_for_models(histories, target_metric='val_loss', window_size=5, min_change=0.001, use_ray=True, num_cpus=None):
    """
    Detect plateaus across multiple models in parallel using Ray.
    
    Args:
        histories: Dictionary of model names to their training histories
        target_metric: The metric to use for measuring convergence
        window_size: Number of epochs to consider for plateau detection
        min_change: Minimum relative change to consider as progress
        use_ray: Whether to use Ray for parallel processing
        num_cpus: Number of CPUs to use for Ray
        
    Returns:
        dict: Dictionary of model names to plateau epochs
    """
    # If only one model or Ray is disabled, use sequential processing
    if len(histories) <= 1 or not use_ray:
        results = {}
        for model_name, history in histories.items():
            plateau_epoch = detect_convergence_plateau(history, target_metric, window_size, min_change)
            results[model_name] = plateau_epoch
        return results
    
    # Try to use Ray for parallel processing
    try:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            init_ray_for_mac(num_cpus)
            
        # Submit tasks to Ray
        tasks = [detect_plateau_ray.remote(model_name, history, target_metric, window_size, min_change) 
                for model_name, history in histories.items()]
        
        # Get results
        model_results = ray.get(tasks)
        return {model_name: plateau_epoch for model_name, plateau_epoch in model_results}
        
    except Exception as e:
        logger.warning(f"Failed to use Ray for parallel processing: {e}. Falling back to sequential processing.")
        # Fallback to sequential processing
        results = {}
        for model_name, history in histories.items():
            plateau_epoch = detect_convergence_plateau(history, target_metric, window_size, min_change)
            results[model_name] = plateau_epoch
        return results


@ray.remote
def compute_model_metrics_ray(model_name, history, metrics_to_compute):
    """Compute various metrics for a single model in parallel"""
    results = {'model_name': model_name}
    
    for metric_name, metric_config in metrics_to_compute.items():
        metric_type = metric_config.get('type')
        metric_args = metric_config.get('args', {})
        
        if metric_type == 'convergence_speed':
            results[metric_name] = calculate_convergence_speed(
                history, 
                target_metric=metric_args.get('target_metric', 'val_loss'),
                threshold=metric_args.get('threshold', 0.1)
            )
        elif metric_type == 'efficiency':
            results[metric_name] = calculate_training_efficiency(
                history, 
                target_metric=metric_args.get('target_metric', 'val_loss'),
                threshold=metric_args.get('threshold', 0.1)
            )
        elif metric_type == 'plateau':
            results[metric_name] = detect_convergence_plateau(
                history, 
                target_metric=metric_args.get('target_metric', 'val_loss'),
                window_size=metric_args.get('window_size', 5),
                min_change=metric_args.get('min_change', 0.001)
            )
    
    return results


def compute_convergence_metrics(histories, metrics_to_compute, use_ray=True, num_cpus=None):
    """
    Compute multiple convergence metrics for multiple models in parallel.
    
    Args:
        histories: Dictionary of model names to their training histories
        metrics_to_compute: Dictionary specifying which metrics to compute
            Example: {
                'time_to_converge': {
                    'type': 'convergence_speed',
                    'args': {'target_metric': 'val_loss', 'threshold': 0.1}
                },
                'training_efficiency': {
                    'type': 'efficiency',
                    'args': {'target_metric': 'val_loss', 'threshold': 0.1}
                },
                'plateau_point': {
                    'type': 'plateau',
                    'args': {'target_metric': 'val_loss', 'window_size': 5}
                }
            }
        use_ray: Whether to use Ray for parallel processing
        num_cpus: Number of CPUs to use for Ray
        
    Returns:
        list: List of dictionaries with metrics for each model
    """
    # If only one model or Ray is disabled, use sequential processing
    if len(histories) <= 1 or not use_ray:
        results = []
        for model_name, history in histories.items():
            model_results = {'model_name': model_name}
            
            for metric_name, metric_config in metrics_to_compute.items():
                metric_type = metric_config.get('type')
                metric_args = metric_config.get('args', {})
                
                if metric_type == 'convergence_speed':
                    model_results[metric_name] = calculate_convergence_speed(
                        history, 
                        target_metric=metric_args.get('target_metric', 'val_loss'),
                        threshold=metric_args.get('threshold', 0.1)
                    )
                elif metric_type == 'efficiency':
                    model_results[metric_name] = calculate_training_efficiency(
                        history, 
                        target_metric=metric_args.get('target_metric', 'val_loss'),
                        threshold=metric_args.get('threshold', 0.1)
                    )
                elif metric_type == 'plateau':
                    model_results[metric_name] = detect_convergence_plateau(
                        history, 
                        target_metric=metric_args.get('target_metric', 'val_loss'),
                        window_size=metric_args.get('window_size', 5),
                        min_change=metric_args.get('min_change', 0.001)
                    )
            
            results.append(model_results)
        
        return results
    
    # Try to use Ray for parallel processing
    try:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            init_ray_for_mac(num_cpus)
            
        # Submit tasks to Ray
        tasks = [compute_model_metrics_ray.remote(model_name, history, metrics_to_compute) 
                for model_name, history in histories.items()]
        
        # Get results
        return ray.get(tasks)
        
    except Exception as e:
        logger.warning(f"Failed to use Ray for parallel processing: {e}. Falling back to sequential processing.")
        # Fallback to sequential processing
        return compute_convergence_metrics(histories, metrics_to_compute, use_ray=False)