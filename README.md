# Lost in Translation - Multiclass Sentiment Analysis

## Team: Lost in Translation
**Dataset:** Sp1786/multiclass-sentiment-analysis-dataset

## Project Overview
This project demonstrates Transfer Learning by fine-tuning a BERT-based model on a multiclass sentiment analysis task. The model is trained to classify text into multiple sentiment categories using PyTorch.

## Dataset
- **Name:** Multiclass Sentiment Analysis Dataset
- **Source:** [Hugging Face - Sp1786/multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset)
- **Task:** Text Classification
- **Classes:** Multiple sentiment categories

## Project Structure
```
.
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── main.ipynb                         # Main notebook with complete workflow
├── src/
│   ├── data_loader.py                # Dataset loading and preprocessing
│   ├── model.py                      # BERT model architecture
│   ├── trainer.py                    # Training loop implementation
│   ├── evaluator.py                  # Evaluation metrics and visualization
│   └── inference.py                  # Inference pipeline
└── outputs/
    ├── confusion_matrix.png          # Confusion matrix visualization
    └── class_distribution.png        # Class distribution plot
```

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Load dataset using Hugging Face `datasets` library
- Visualize class distribution in training set
- Analyze class imbalance (if present)

### 2. Model Fine-tuning
- Base Model: BERT (bert-base-uncased)
- Framework: PyTorch (no HuggingFace Trainer)
- Custom training loop implementation
- Optimization: AdamW optimizer
- Learning rate scheduling

### 3. Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision across all classes
- **Recall**: Weighted recall across all classes
- **F1-Score**: Weighted F1-score
- **Confusion Matrix**: Visual representation of predictions

### 4. Inference Pipeline
Function: `predict_text(text: str) -> dict`
- Input: Raw text string
- Output: Predicted class label and confidence score
- Tested with 5 custom examples

## Requirements
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
```

## How to Run

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the main notebook:**
Open `main.ipynb` and execute all cells sequentially.

3. **Or run individual components:**
```python
# Load data
from src.data_loader import load_data
train_loader, val_loader, test_loader, num_classes = load_data()

# Train model
from src.trainer import train_model
model = train_model(train_loader, val_loader, num_classes)

# Evaluate
from src.evaluator import evaluate_model
evaluate_model(model, test_loader)

# Inference
from src.inference import predict_text
result = predict_text("This product is amazing!", model)
```

## Results
*(Results will be populated after training)*

### Performance Metrics
- Accuracy: TBD
- Precision (weighted): TBD
- Recall (weighted): TBD
- F1-Score (weighted): TBD

### Sample Predictions
*(Sample predictions will be added after inference testing)*

## Key Implementation Details
- **No HuggingFace Trainer**: Custom PyTorch training loop
- **Pure PyTorch**: All training logic implemented from scratch
- **Modular Design**: Separate modules for data loading, training, evaluation, and inference
- **Reproducibility**: Fixed random seeds for consistent results

## Submission Details
- **Team Name:** Lost in Translation
- **Dataset:** multiclass-sentiment-analysis-dataset
- **Repository Name:** LostInTranslation_multiclass-sentiment-analysis-dataset
- **Deadline:** Monday, Feb 16th, 10 AM

## Authors
Lost in Translation Team

## License
MIT License
