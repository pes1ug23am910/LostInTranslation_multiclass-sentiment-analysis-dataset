"""
Inference pipeline for making predictions on new text.
This module provides functions for making predictions with the trained model.
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from typing import Dict, List, Tuple
import numpy as np


class InferencePipeline:
    """
    Inference pipeline for text classification.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer (BERT tokenizer)
        class_names: Names of the classes
        device: Device to run inference on
        max_length: Maximum sequence length
    """
    
    def __init__(self, model, tokenizer, class_names: List[str], 
                 device: str = None, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.max_length = max_length
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict_text(self, text: str) -> Dict[str, any]:
        """
        Predict the class label for a single text.
        
        Args:
            text: Input text string
        
        Returns:
            Dictionary containing predicted label, confidence score, and all class probabilities
        """
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get prediction
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
        
        # Prepare result
        result = {
            'text': text,
            'predicted_label': self.class_names[predicted_class],
            'predicted_class_id': predicted_class,
            'confidence': confidence,
            'all_probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(all_probs)
            }
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict class labels for a batch of texts.
        
        Args:
            texts: List of input text strings
        
        Returns:
            List of dictionaries containing predictions for each text
        """
        results = []
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            token_type_ids = encodings.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            
            # Make predictions
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Get probabilities
                probabilities = F.softmax(logits, dim=1)
                confidences, predicted_classes = torch.max(probabilities, dim=1)
                
                # Convert to numpy
                predicted_classes = predicted_classes.cpu().numpy()
                confidences = confidences.cpu().numpy()
                all_probs = probabilities.cpu().numpy()
            
            # Prepare results
            for j, text in enumerate(batch_texts):
                pred_class = predicted_classes[j]
                confidence = confidences[j]
                probs = all_probs[j]
                
                result = {
                    'text': text,
                    'predicted_label': self.class_names[pred_class],
                    'predicted_class_id': int(pred_class),
                    'confidence': float(confidence),
                    'all_probabilities': {
                        self.class_names[k]: float(prob) 
                        for k, prob in enumerate(probs)
                    }
                }
                results.append(result)
        
        return results


def create_inference_pipeline(model, tokenizer, class_names: List[str],
                             device: str = None, max_length: int = 128) -> InferencePipeline:
    """
    Create an inference pipeline.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        class_names: Names of the classes
        device: Device to run inference on
        max_length: Maximum sequence length
    
    Returns:
        InferencePipeline instance
    """
    return InferencePipeline(model, tokenizer, class_names, device, max_length)


def print_prediction(result: Dict[str, any], show_all_probs: bool = False):
    """
    Pretty print prediction result.
    
    Args:
        result: Prediction result dictionary
        show_all_probs: Whether to show all class probabilities
    """
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Text: {result['text']}")
    print("-"*60)
    print(f"Predicted Label: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    
    if show_all_probs:
        print("-"*60)
        print("All Class Probabilities:")
        sorted_probs = sorted(result['all_probabilities'].items(), 
                             key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            bar = '█' * int(prob * 50)
            print(f"  {class_name:20s}: {prob:.4f} {bar}")
    
    print("="*60 + "\n")


def test_custom_examples(pipeline: InferencePipeline, examples: List[str]):
    """
    Test the inference pipeline with custom examples.
    
    Args:
        pipeline: InferencePipeline instance
        examples: List of example texts
    """
    print("\n" + "="*60)
    print("TESTING CUSTOM EXAMPLES")
    print("="*60 + "\n")
    
    for i, text in enumerate(examples, 1):
        print(f"Example {i}:")
        result = pipeline.predict_text(text)
        print_prediction(result, show_all_probs=True)
