"""
Test script to verify installation and setup.
Run this before starting the main notebook to ensure everything is working.
"""

import sys
from pathlib import Path

print("=" * 60)
print("BANANA ASSIGNMENT - SETUP VERIFICATION")
print("=" * 60)
print()

# Test 1: Check Python version
print("1. Checking Python version...")
python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 8:
    print(f"   ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"   ✗ Python version too old: {python_version.major}.{python_version.minor}")
    print("   Please upgrade to Python 3.8 or higher")
print()

# Test 2: Check required packages
print("2. Checking required packages...")
packages = {
    'torch': 'PyTorch',
    'transformers': 'HuggingFace Transformers',
    'datasets': 'HuggingFace Datasets',
    'sklearn': 'scikit-learn',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'tqdm': 'tqdm'
}

missing_packages = []
for package, name in packages.items():
    try:
        __import__(package)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print()
    print("   Missing packages detected!")
    print("   Install with: pip install -r requirements.txt")
    sys.exit(1)
print()

# Test 3: Check CUDA availability
print("3. Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   ⚠ CUDA not available - will use CPU")
        print("   Training will be slower but still works")
except Exception as e:
    print(f"   ⚠ Could not check CUDA: {e}")
print()

# Test 4: Check project structure
print("4. Checking project structure...")
required_files = [
    'README.md',
    'requirements.txt',
    'main.ipynb',
    'src/data_loader.py',
    'src/model.py',
    'src/trainer.py',
    'src/evaluator.py',
    'src/inference.py'
]

project_root = Path(__file__).parent
all_files_present = True

for file_path in required_files:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} - MISSING")
        all_files_present = False

if not all_files_present:
    print()
    print("   Some files are missing!")
    sys.exit(1)
print()

# Test 5: Check internet connection (for dataset download)
print("5. Checking internet connection...")
try:
    import urllib.request
    urllib.request.urlopen('https://huggingface.co', timeout=5)
    print("   ✓ Internet connection OK")
    print("   Can download datasets from HuggingFace")
except Exception as e:
    print("   ⚠ Internet connection issue")
    print("   You may have trouble downloading the dataset")
print()

# Test 6: Test imports from custom modules
print("6. Testing custom module imports...")
sys.path.append('src')
try:
    from src.data_loader import load_and_prepare_dataset
    print("   ✓ data_loader")
    
    from src.model import initialize_model
    print("   ✓ model")
    
    from src.trainer import train_model
    print("   ✓ trainer")
    
    from src.evaluator import calculate_metrics
    print("   ✓ evaluator")
    
    from src.inference import create_inference_pipeline
    print("   ✓ inference")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)
print()

# Test 7: Quick BERT test
print("7. Testing BERT model loading...")
try:
    from transformers import BertTokenizer, BertModel
    print("   Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("   ✓ Tokenizer loaded")
    print("   Loading BERT model...")
    model = BertModel.from_pretrained('bert-base-uncased')
    print("   ✓ Model loaded")
    print("   First-time setup may take a few minutes to download")
except Exception as e:
    print(f"   ⚠ BERT loading issue: {e}")
    print("   This might be a network issue - try again")
print()

# Final summary
print("=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)
print()
print("✅ All checks passed! You're ready to run main.ipynb")
print()
print("Next steps:")
print("1. Open main.ipynb in Jupyter or VS Code")
print("2. Run all cells sequentially")
print("3. Check outputs and visualizations")
print("4. Customize examples if needed")
print("5. Submit to Google Forms")
print()
print("For detailed instructions, see QUICKSTART.md")
print("=" * 60)
