#!/usr/bin/env python
"""Training utilities for model training and evaluation."""

import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple, Optional, List


def create_optimizer(model, config):
    """Create optimizer based on config."""
    optimizer_type = config["hyperparameters"].get("optimizer", "adamw").lower()
    lr = config["hyperparameters"].get("lr", 0.001)
    weight_decay = config["hyperparameters"].get("weight_decay", 0.0)
    
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        momentum = config["hyperparameters"].get("momentum", 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer type {optimizer_type}, using AdamW")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config."""
    if not config["hyperparameters"].get("use_scheduler", False):
        return None
        
    scheduler_type = config["hyperparameters"].get("scheduler_type", "cosine").lower()
    epochs = config["hyperparameters"].get("epochs", 100)
    
    if scheduler_type == "step":
        step_size = config["hyperparameters"].get("scheduler_step_size", 30)
        gamma = config["hyperparameters"].get("scheduler_gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "plateau":
        patience = config["hyperparameters"].get("scheduler_patience", 10)
        factor = config["hyperparameters"].get("scheduler_factor", 0.5)
        min_lr = config["hyperparameters"].get("min_lr", 1e-7)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=factor, patience=patience, min_lr=min_lr
        )
    elif scheduler_type == "cosine":
        min_lr = config["hyperparameters"].get("min_lr", 1e-7)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
    elif scheduler_type == "warmup_cosine":
        min_lr = config["hyperparameters"].get("min_lr", 1e-7)
        warmup_epochs = config["hyperparameters"].get("warmup_epochs", 5)
        return get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_epochs,
            num_training_steps=epochs,
            min_lr=min_lr
        )
    else:
        logger.warning(f"Unknown scheduler type {scheduler_type}, not using a scheduler")
        return None


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
    """
    Create a schedule with linear warmup and cosine annealing.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, history, config, checkpoint_path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': history,
        'config': config
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} not found, starting from scratch")
        return 0, None, None
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        history = checkpoint.get('history', None)
        config = checkpoint.get('config', None)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], history, config
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0, None, None


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for regression.
    """
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # R² score
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def train_model(
    model: nn.Module,
    t_train: np.ndarray,
    data_train: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    validation_split: float = 0.1,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    start_epoch: int = 0
) -> Dict[str, List]:
    """Train a model on the given data."""
    tensor_t = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1).to(device)
    tensor_data = torch.tensor(data_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    dataset = TensorDataset(tensor_t, tensor_data)
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    batch_size = config["hyperparameters"].get("batch_size", 32)
    num_workers = config["hyperparameters"].get("num_workers", 2)
    drop_last = config["hyperparameters"].get("drop_last", True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    if optimizer is None:
        optimizer = create_optimizer(model, config)
    
    if scheduler is None and config["hyperparameters"].get("use_scheduler", False):
        scheduler = create_scheduler(optimizer, config)
    
    epochs = config["hyperparameters"].get("epochs", 100)
    clip_gradients = config["hyperparameters"].get("clip_gradients", False)
    clip_value = config["hyperparameters"].get("clip_value", 1.0)
    early_stopping = config["hyperparameters"].get("early_stopping", False)
    patience = config["hyperparameters"].get("early_stopping_patience", 10)
    min_delta = config["hyperparameters"].get("early_stopping_min_delta", 0.0001)
    save_checkpoints = config["hyperparameters"].get("save_checkpoints", False)
    checkpoint_dir = config["hyperparameters"].get("checkpoint_dir", "models/checkpoints")
    
    criterion = nn.MSELoss()
    
    if hasattr(model, 'history') and isinstance(model.history, dict):
        history = model.history
        if start_epoch > 0:
            if 'train_loss' in history and len(history['train_loss']) < start_epoch:
                history['train_loss'] = history['train_loss'] + [0] * (start_epoch - len(history['train_loss']))
            if 'val_loss' in history and len(history['val_loss']) < start_epoch:
                history['val_loss'] = history['val_loss'] + [0] * (start_epoch - len(history['val_loss']))
            if 'epochs' in history and len(history['epochs']) < start_epoch:
                history['epochs'] = list(range(1, start_epoch + 1))
            if 'metrics' in history and len(history['metrics']) < start_epoch:
                history['metrics'] = history['metrics'] + [{}] * (start_epoch - len(history['metrics']))
    else:
        history = {
            "train_loss": [],
            "val_loss": [],
            "epochs": [],
            "metrics": []
        }

    best_val_loss = float('inf')
    no_improve_epochs = 0

    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
    

    grad_accum_steps = config["hyperparameters"].get("gradient_accumulation_steps", 1)

    use_amp = config["hyperparameters"].get("use_amp", False)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    nan_detection = config["hyperparameters"].get("nan_detection", True)
    
    logger.info(f"Starting training for {epochs} epochs (resuming from epoch {start_epoch + 1})")
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        optimizer.zero_grad()  # Zero gradients at the beginning of the epoch

        for i, (x_batch, y_batch) in enumerate(train_loader):

            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss = loss / grad_accum_steps
            else:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss = loss / grad_accum_steps
            
            if use_amp and scaler:
                scaler.scale(loss).backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    if clip_gradients:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    if clip_gradients:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * grad_accum_steps
            
            if nan_detection and torch.isnan(loss):
                logger.warning(f"NaN detected in training loss at epoch {epoch + 1}, batch {i + 1}")
                if i > 0:
                    break
        
        train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                
                if nan_detection and torch.isnan(loss):
                    logger.warning(f"NaN detected in validation loss at epoch {epoch + 1}")
                    break
        
        val_loss = val_loss / len(val_loader)
        
        all_preds = np.vstack(all_preds).reshape(-1)
        all_targets = np.vstack(all_targets).reshape(-1)
        metrics = compute_metrics(all_targets, all_preds)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch + 1)
        history["metrics"].append(metrics)
        
        if hasattr(model, 'history'):
            model.history = history
        

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"rmse={metrics['rmse']:.6f}, r2={metrics['r2']:.6f}, "
            f"lr={current_lr:.8f}, time={(time.time() - start_time):.2f}s"
        )
        
        if save_checkpoints:
            if val_loss < best_val_loss:
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, history, config,
                    os.path.join(checkpoint_dir, "best_model.pt")
                )
            
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, history, config,
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                )
        
        if early_stopping:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    if save_checkpoints:
        save_checkpoint(
            model, optimizer, scheduler, epochs, history, config,
            os.path.join(checkpoint_dir, f"checkpoint_final.pt")
        )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    history["training_time"] = training_time
    
    if hasattr(model, 'history'):
        model.history = history
    
    return history