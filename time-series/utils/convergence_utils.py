"""Utilities for measuring model convergence."""

import numpy as np
import time
from loguru import logger


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


def compare_convergence(histories, target_metric='val_loss', threshold=0.1):
    """
    Compare convergence speed between multiple models.
    
    Args:
        histories: Dictionary of model names to their training histories
        target_metric: The metric to use for measuring convergence (default: 'val_loss')
        threshold: Relative threshold for determining convergence as a fraction (default: 0.1 = 10%)
        
    Returns:
        dict: Dictionary containing convergence comparison results
    """
    results = {}
    
    for model_name, history in histories.items():
        efficiency = calculate_training_efficiency(history, target_metric, threshold)
        results[model_name] = efficiency
    
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