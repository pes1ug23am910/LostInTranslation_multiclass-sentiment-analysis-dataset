"""
Data loading and preprocessing utilities for BERT fine-tuning.
This module handles dataset loading, tokenization, and DataLoader creation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
import numpy as np
from typing import Tuple, Dict, Any


class SentimentDataset(Dataset):
    """
    Custom PyTorch Dataset for sentiment analysis.
    
    Args:
        encodings: Tokenized text encodings
        labels: Sentiment labels
    """
    def __init__(self, encodings: Dict[str, Any], labels: list):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.labels)


def load_and_prepare_dataset(dataset_name: str = "Sp1786/multiclass-sentiment-analysis-dataset",
                             tokenizer_name: str = "bert-base-uncased",
                             max_length: int = 128) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    """
    Load dataset from Hugging Face and prepare it for training.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Tuple containing train, validation, and test datasets, and metadata
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # Get the text and label column names (they vary by dataset)
    # We'll inspect the dataset to find the correct column names
    sample = dataset['train'][0]
    text_col = None
    label_col = None
    
    # Common text column names
    for col in ['text', 'sentence', 'content', 'review', 'comment']:
        if col in sample:
            text_col = col
            break
    
    # Common label column names
    for col in ['label', 'labels', 'sentiment', 'category']:
        if col in sample:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        raise ValueError(f"Could not identify text and label columns. Available columns: {list(sample.keys())}")
    
    print(f"Using text column: '{text_col}' and label column: '{label_col}'")
    
    # Extract texts and labels
    train_texts = dataset['train'][text_col]
    train_labels = dataset['train'][label_col]
    
    # Check if validation set exists, otherwise split from train
    if 'validation' in dataset:
        val_texts = dataset['validation'][text_col]
        val_labels = dataset['validation'][label_col]
    else:
        # Split train into train and validation (80-20)
        split_idx = int(0.8 * len(train_texts))
        val_texts = train_texts[split_idx:]
        val_labels = train_labels[split_idx:]
        train_texts = train_texts[:split_idx]
        train_labels = train_labels[:split_idx]
    
    # Check if test set exists
    if 'test' in dataset:
        test_texts = dataset['test'][text_col]
        test_labels = dataset['test'][label_col]
    else:
        # Use validation as test
        test_texts = val_texts
        test_labels = val_labels
    
    # Tokenize texts
    print("Tokenizing texts...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, 
                                max_length=max_length, return_tensors=None)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, 
                             max_length=max_length, return_tensors=None)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, 
                               max_length=max_length, return_tensors=None)
    
    # Create datasets
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)
    
    # Calculate metadata
    num_classes = len(set(train_labels))
    class_names = sorted(set(train_labels))
    
    metadata = {
        'num_classes': num_classes,
        'class_names': class_names,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'text_column': text_col,
        'label_column': label_col,
        'tokenizer': tokenizer
    }
    
    print(f"Dataset prepared successfully!")
    print(f"  Train size: {metadata['train_size']}")
    print(f"  Validation size: {metadata['val_size']}")
    print(f"  Test size: {metadata['test_size']}")
    print(f"  Number of classes: {num_classes}")
    
    return train_dataset, val_dataset, test_dataset, metadata


def create_data_loaders(train_dataset: Dataset, 
                       val_dataset: Dataset, 
                       test_dataset: Dataset,
                       batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for DataLoaders
    
    Returns:
        Tuple containing train, validation, and test DataLoaders
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def get_class_distribution(dataset: Dataset) -> Dict[int, int]:
    """
    Calculate class distribution in a dataset.
    
    Args:
        dataset: PyTorch Dataset
    
    Returns:
        Dictionary mapping class labels to counts
    """
    labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))
