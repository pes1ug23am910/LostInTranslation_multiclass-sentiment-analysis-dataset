"""
Evaluation metrics and visualization utilities.
This module provides functions for computing metrics and creating visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import pandas as pd


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary containing accuracy, precision, recall, and F1-score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (weighted)")
    print(f"Recall:    {metrics['recall']:.4f} (weighted)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} (weighted)")
    print("="*60 + "\n")


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: List[str] = None,
                         save_path: str = None,
                         figsize: Tuple[int, int] = (10, 8)):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        save_path: Path to save the plot
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_class_distribution(class_distribution: Dict[int, int],
                           class_names: List[str] = None,
                           save_path: str = None,
                           figsize: Tuple[int, int] = (10, 6)):
    """
    Plot class distribution.
    
    Args:
        class_distribution: Dictionary mapping class labels to counts
        class_names: Names of the classes
        save_path: Path to save the plot
        figsize: Figure size
    """
    labels = sorted(class_distribution.keys())
    counts = [class_distribution[label] for label in labels]
    
    if class_names:
        label_names = [class_names[i] if i < len(class_names) else f"Class {i}" 
                      for i in labels]
    else:
        label_names = [f"Class {i}" for i in labels]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create bar plot
    bars = plt.bar(range(len(labels)), counts, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Class Distribution in Training Set', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(range(len(labels)), label_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()
    
    # Check for class imbalance
    total = sum(counts)
    print("\nClass Distribution Analysis:")
    print("-" * 60)
    for label, count, name in zip(labels, counts, label_names):
        percentage = (count / total) * 100
        print(f"{name}: {count:,} samples ({percentage:.2f}%)")
    print("-" * 60)
    
    # Imbalance detection
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        print("⚠️  Significant class imbalance detected!")
        print("   Consider using techniques like:")
        print("   - Class weights in loss function")
        print("   - Oversampling minority classes")
        print("   - Undersampling majority classes")
    elif imbalance_ratio > 1.5:
        print("⚠️  Moderate class imbalance detected.")
    else:
        print("✓ Classes are relatively balanced.")


def plot_training_history(history: Dict[str, List], 
                          save_path: str = None,
                          figsize: Tuple[int, int] = (14, 5)):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        figsize: Figure size
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def print_classification_report(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               class_names: List[str] = None):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
    """
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("="*60 + "\n")


def analyze_predictions(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       texts: List[str] = None,
                       class_names: List[str] = None,
                       num_examples: int = 5):
    """
    Analyze and display sample predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: Original texts (optional)
        class_names: Names of the classes
        num_examples: Number of examples to show
    """
    # Find correct and incorrect predictions
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask
    
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]
    
    print("\n" + "="*60)
    print(f"PREDICTION ANALYSIS")
    print("="*60)
    print(f"Total predictions: {len(y_true)}")
    print(f"Correct predictions: {correct_mask.sum()} ({100*correct_mask.mean():.2f}%)")
    print(f"Incorrect predictions: {incorrect_mask.sum()} ({100*incorrect_mask.mean():.2f}%)")
    print("="*60 + "\n")
    
    if texts and len(incorrect_indices) > 0:
        print("Sample Incorrect Predictions:")
        print("-" * 60)
        
        num_show = min(num_examples, len(incorrect_indices))
        sample_indices = np.random.choice(incorrect_indices, num_show, replace=False)
        
        for idx in sample_indices:
            true_label = class_names[y_true[idx]] if class_names else y_true[idx]
            pred_label = class_names[y_pred[idx]] if class_names else y_pred[idx]
            text_preview = texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx]
            
            print(f"\nText: {text_preview}")
            print(f"True Label: {true_label}")
            print(f"Predicted: {pred_label}")
            print("-" * 60)
