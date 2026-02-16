# Quick Start Guide - Banana Assignment

## Overview
This guide will help you complete the Banana Assignment step by step.

## What's Included

### Project Structure
```
.
├── README.md                  # Complete project documentation
├── requirements.txt           # Python dependencies
├── main.ipynb                # Main notebook (YOUR PRIMARY WORK FILE)
├── .gitignore                # Git ignore rules
├── src/                      # Utility modules
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── model.py              # BERT classifier architecture
│   ├── trainer.py            # Custom PyTorch training loop
│   ├── evaluator.py          # Metrics and visualization
│   └── inference.py          # Prediction pipeline
└── outputs/                  # Generated visualizations
```

## Step-by-Step Instructions

### 1. Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```

**Required packages:**
- torch (PyTorch)
- transformers (BERT model & tokenizer)
- datasets (Hugging Face datasets)
- scikit-learn (metrics)
- matplotlib, seaborn (visualization)
- tqdm (progress bars)

### 2. Open and Run the Notebook
1. Open `main.ipynb` in Jupyter or VS Code
2. **Run all cells sequentially** (Shift + Enter for each cell)
3. The notebook is already complete and will:
   - Load the dataset
   - Perform EDA with visualizations
   - Fine-tune BERT with PyTorch
   - Evaluate with all required metrics
   - Create inference pipeline
   - Test on 5 custom examples

### 3. Customize (Optional)
You can modify these parameters in the notebook:

**Dataset Selection** (Cell 2.1):
```python
dataset_name = "Sp1786/multiclass-sentiment-analysis-dataset"
# Change to another dataset if needed
```

**Training Parameters** (Cell 3.3):
```python
num_epochs = 3          # Increase for better accuracy
batch_size = 16         # Adjust based on GPU memory
learning_rate = 2e-5    # Try 3e-5 or 5e-5
```

**Custom Examples** (Cell 5.1):
```python
custom_examples = [
    "Your own test text 1",
    "Your own test text 2",
    # ... add 3-5 examples
]
```

### 4. Expected Outputs

After running the notebook, you'll get:

#### Visualizations (saved in `outputs/`):
- `class_distribution.png` - Training set class distribution
- `confusion_matrix.png` - Model predictions confusion matrix
- `training_history.png` - Training/validation loss and accuracy curves

#### Model Files (saved in `models/`):
- `bert_sentiment_classifier.pt` - Trained model weights

#### Evaluation Metrics (printed):
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Detailed classification report

## Understanding the Code

### 1. EDA Section (Cells 2.x)
- **Purpose:** Load dataset and understand class distribution
- **Key Function:** `load_and_prepare_dataset()` handles everything
- **Output:** Bar chart showing class distribution with imbalance analysis

### 2. Fine-tuning Section (Cells 3.x)
- **Purpose:** Train BERT using pure PyTorch
- **Key Components:**
  - `BERTClassifier` - Custom model class
  - `train_model()` - Complete training loop
  - AdamW optimizer with learning rate scheduling
- **Important:** NO HuggingFace Trainer used (as required)

### 3. Evaluation Section (Cells 4.x)
- **Purpose:** Test model and calculate metrics
- **Key Functions:**
  - `evaluate()` - Get predictions on test set
  - `calculate_metrics()` - Compute all required metrics
  - `plot_confusion_matrix()` - Visualize predictions

### 4. Inference Section (Cells 5.x)
- **Purpose:** Create prediction pipeline
- **Key Function:** `predict_text(text: str)`
- **Usage:**
```python
result = predict_text("Your text here")
print(result['predicted_label'])
print(result['confidence'])
```

## Troubleshooting

### Issue: Out of Memory Error
**Solution:** Reduce batch size in Cell 3.1:
```python
batch_size = 8  # or even 4
```

### Issue: Dataset Not Loading
**Solution:** Check internet connection. HuggingFace needs to download the dataset first.

### Issue: Import Errors
**Solution:** Make sure you're in the right directory:
```python
import sys
sys.path.append('src')  # Should be in notebook
```

### Issue: CUDA Out of Memory
**Solution:** 
1. Reduce batch size
2. Use CPU instead: Set `device = 'cpu'` in Cell 1
3. Reduce max_length: `max_length=64` instead of 128

## Training Tips

### Fast Training (for testing):
```python
num_epochs = 1
batch_size = 32
```

### Better Accuracy (more time):
```python
num_epochs = 5
batch_size = 16
learning_rate = 3e-5
```

### Limited GPU Memory:
```python
batch_size = 8
max_length = 64
```

## Submission Checklist

Before submitting, ensure you have:

- [ ] All cells in `main.ipynb` executed successfully
- [ ] EDA section shows class distribution plot
- [ ] Model training completed (all epochs)
- [ ] Evaluation metrics displayed (Accuracy, Precision, Recall, F1)
- [ ] Confusion matrix generated
- [ ] `predict_text()` function working
- [ ] 5 custom examples tested
- [ ] All helper functions visible in `src/` folder
- [ ] README.md properly filled out
- [ ] Code pushed to GitHub repository
- [ ] Repository named: `LostInTranslation_multiclass-sentiment-analysis-dataset`

## GitHub Submission

### Initialize Git (if not done):
```bash
git init
git add .
git commit -m "Initial commit - BERT fine-tuning project"
```

### Push to GitHub:
```bash
git remote add origin YOUR_REPO_URL
git branch -M main
git push -u origin main
```

### Repository Name Format:
```
TeamName_dataset-picked
Example: LostInTranslation_multiclass-sentiment-analysis-dataset
```

## Key Requirements Met

✅ **EDA:** Loaded dataset, visualized class distribution, commented on imbalance  
✅ **Fine-tuning:** Pure PyTorch (no HuggingFace Trainer)  
✅ **Metrics:** Accuracy, Precision, Recall, F1-Score (weighted)  
✅ **Confusion Matrix:** Visual representation of predictions  
✅ **Inference:** `predict_text(text: str)` function created  
✅ **Testing:** 5 custom examples tested  
✅ **Code Visibility:** All helper functions in `src/` folder  
✅ **Documentation:** Clear README.md  

## Time Estimate

- **Setup & Installation:** 10-15 minutes
- **Running Notebook:** 30-60 minutes (depends on hardware)
- **Customization & Testing:** 15-30 minutes
- **Documentation & Submission:** 15-20 minutes

**Total:** ~1.5-2.5 hours

## Need Help?

1. Check error messages in the notebook output
2. Review the README.md for detailed explanations
3. Check the source code in `src/` folder
4. All functions are well-documented with docstrings

## Good Luck! 🚀

Remember:
- Run cells sequentially
- Don't skip the EDA section
- Save your work frequently (Ctrl+S)
- Check all outputs before submission
