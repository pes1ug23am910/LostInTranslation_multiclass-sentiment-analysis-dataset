"""
Configuration file for the project.
Contains all hyperparameters and settings in one place.
"""

# Dataset Configuration
DATASET_CONFIG = {
    'name': 'Sp1786/multiclass-sentiment-analysis-dataset',
    'max_length': 128,
    'test_size': 0.2,
    'val_size': 0.1,
}

# Model Configuration
MODEL_CONFIG = {
    'name': 'bert-base-uncased',
    'dropout': 0.3,
    'hidden_size': 768,
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 3,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'seed': 42,
}

# Paths
PATHS = {
    'model_save_path': 'models/bert_sentiment_classifier.pt',
    'output_dir': 'outputs',
    'confusion_matrix': 'outputs/confusion_matrix.png',
    'class_distribution': 'outputs/class_distribution.png',
    'training_history': 'outputs/training_history.png',
}

# Inference Configuration
INFERENCE_CONFIG = {
    'max_length': 128,
    'batch_size': 32,
}

# Device Configuration
DEVICE_CONFIG = {
    'use_cuda': True,  # Set to False to force CPU
    'cuda_device': 0,
}


def get_device():
    """Get the device to use for training/inference."""
    import torch
    
    if DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available():
        device = f"cuda:{DEVICE_CONFIG['cuda_device']}"
    else:
        device = 'cpu'
    
    return device


def print_config():
    """Print all configuration settings."""
    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)
    print("\n📊 Dataset:")
    for key, value in DATASET_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n🤖 Model:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n🏋️ Training:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n💾 Paths:")
    for key, value in PATHS.items():
        print(f"  {key}: {value}")
    
    print("\n🔮 Inference:")
    for key, value in INFERENCE_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n💻 Device:")
    print(f"  Using: {get_device()}")
    
    print("=" * 60)


if __name__ == "__main__":
    print_config()
