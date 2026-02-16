"""
BERT model architecture for text classification.
This module defines the BERT-based classifier using PyTorch.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    """
    BERT-based text classifier.
    
    This model uses a pre-trained BERT model as the backbone and adds
    a classification head on top for multi-class classification.
    
    Args:
        num_classes: Number of output classes
        model_name: Name of the pre-trained BERT model
        dropout: Dropout probability for the classification head
    """
    
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", dropout: float = 0.3):
        super(BERTClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Get the hidden size from BERT config
        self.hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # For storing attention weights if needed
        self.num_classes = num_classes
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                token_type_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
        
        Returns:
            Logits for each class [batch_size, num_classes]
        """
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Extract [CLS] token representation (first token)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        return logits
    
    def freeze_bert_encoder(self):
        """
        Freeze the BERT encoder parameters.
        Only the classification head will be trained.
        """
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        """
        Unfreeze the BERT encoder parameters.
        The entire model will be trained.
        """
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_layers(self, n: int = 2):
        """
        Unfreeze only the last n layers of BERT encoder.
        
        Args:
            n: Number of last layers to unfreeze
        """
        # Freeze all parameters first
        self.freeze_bert_encoder()
        
        # Unfreeze the last n layers
        for layer in self.bert.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Always unfreeze the pooler
        for param in self.bert.pooler.parameters():
            param.requires_grad = True


def initialize_model(num_classes: int, 
                     model_name: str = "bert-base-uncased", 
                     dropout: float = 0.3,
                     device: str = None) -> BERTClassifier:
    """
    Initialize the BERT classifier model.
    
    Args:
        num_classes: Number of output classes
        model_name: Name of the pre-trained BERT model
        dropout: Dropout probability
        device: Device to load the model on (defaults to cuda if available)
    
    Returns:
        Initialized BERTClassifier model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Initializing model: {model_name}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {device}")
    
    model = BERTClassifier(num_classes, model_name, dropout)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model
