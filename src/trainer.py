"""
Training loop implementation for BERT fine-tuning.
This module handles the complete training process using pure PyTorch.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict, List
import time


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                scheduler,
                criterion: nn.Module,
                device: str,
                epoch: int) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Check if token_type_ids exists
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids=input_ids, 
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        
        # Update progress bar
        total_loss += loss.item()
        avg_loss = total_loss / (progress_bar.n + 1)
        accuracy = correct_predictions / total_predictions
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{accuracy:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy


def evaluate(model: nn.Module,
            data_loader: DataLoader,
            criterion: nn.Module,
            device: str,
            desc: str = "Eval") -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
        desc: Description for progress bar
    
    Returns:
        Tuple of (average loss, accuracy, all predictions, all labels)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc=desc, leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Check if token_type_ids exists
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            # Forward pass
            logits = model(input_ids=input_ids, 
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


def train_model(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               num_epochs: int = 3,
               learning_rate: float = 2e-5,
               device: str = None,
               warmup_steps: int = 0) -> Tuple[nn.Module, Dict[str, List]]:
    """
    Complete training loop for the model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        warmup_steps: Number of warmup steps for learning rate
    
    Returns:
        Tuple of (trained model, training history)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"{'='*60}\n")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    if warmup_steps == 0:
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    # Create warmup scheduler followed by linear decay
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, 
                                total_iters=warmup_steps)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                              total_iters=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, 
                            schedulers=[warmup_scheduler, decay_scheduler],
                            milestones=[warmup_steps])
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, desc=f"Epoch {epoch} [Val]"
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best model! (Val Acc: {best_val_acc:.4f})")
        print()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation accuracy: {best_val_acc:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}\n")
    
    return model, history


def save_model(model: nn.Module, path: str):
    """
    Save model to disk.
    
    Args:
        model: Model to save
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: str = None) -> nn.Module:
    """
    Load model from disk.
    
    Args:
        model: Model architecture (initialized)
        path: Path to load the model from
        device: Device to load the model on
    
    Returns:
        Loaded model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model
