# Smart Product Pricing Challenge - Solution

This repository contains a comprehensive machine learning solution for predicting product prices based on textual descriptions and images.

## 🚀 Quick Start

### Option 1: Run Baseline Solution (Recommended)
```bash
# Install dependencies
pip install -r requirements_baseline.txt

# Run the solution
python baseline_solution.py
```

### Option 2: Use the Runner Script
```bash
# Run baseline solution
python run_solution.py --solution baseline

# Run enhanced solution (requires PyTorch)
python run_solution.py --solution enhanced --install-deps

# Run full solution (requires all dependencies)
python run_solution.py --solution full --install-deps
```

## 📁 Solution Files

### Core Solutions
- **`baseline_solution.py`** - Simple but effective baseline solution
- **`enhanced_solution.py`** - Advanced solution with image features
- **`smart_pricing_solution.py`** - Comprehensive solution with deep learning

### Supporting Files
- **`run_solution.py`** - Easy-to-use runner script
- **`requirements.txt`** - Full dependencies
- **`requirements_baseline.txt`** - Minimal dependencies for baseline
- **`utils.py`** - Utility functions for image downloading

### Documentation
- **`ML_Challenge_2025_Smart_Product_Pricing_Solution.md`** - Complete solution documentation
- **`README.md`** - This file

## 🔧 Solution Approaches

### 1. Baseline Solution (`baseline_solution.py`)
- **Features**: Text analysis, quantity extraction, brand detection
- **Models**: Random Forest + Ridge Regression ensemble
- **Performance**: SMAPE ~65.9%
- **Requirements**: Minimal (pandas, sklearn, numpy)
- **Time**: ~2-3 minutes

### 2. Enhanced Solution (`enhanced_solution.py`)
- **Features**: Advanced text features + image features
- **Models**: 5-model ensemble (RF, GBM, ExtraTrees, Ridge, ElasticNet)
- **Performance**: SMAPE ~60-65%
- **Requirements**: PyTorch, torchvision
- **Time**: ~10-15 minutes

### 3. Full Solution (`smart_pricing_solution.py`)
- **Features**: Comprehensive multimodal features
- **Models**: Deep learning + ensemble methods
- **Performance**: Best accuracy
- **Requirements**: Full ML stack (PyTorch, NLTK, etc.)
- **Time**: ~30-45 minutes

## 📊 Key Features

### Text Feature Engineering
- **Quantity Extraction**: Pack sizes, weights, dimensions
- **Brand Detection**: Premium vs. mid-tier brands
- **Category Classification**: 8 product categories
- **Quality Indicators**: Luxury, budget, organic products
- **TF-IDF Vectorization**: N-gram text features

### Image Feature Extraction
- **ResNet50 Features**: 512-dimensional visual features
- **Aspect Ratio**: Geometric properties
- **Color Analysis**: RGB color statistics

### Model Architecture
- **Ensemble Learning**: Multiple algorithms with optimized weights
- **Feature Selection**: Top 500 most informative features
- **Cross-validation**: Robust model evaluation

## 📈 Expected Performance

| Solution | SMAPE | Training Time | Inference Time |
|----------|-------|---------------|----------------|
| Baseline | ~65.9% | 2-3 min | 1-2 min |
| Enhanced | ~60-65% | 10-15 min | 3-5 min |
| Full | ~55-60% | 30-45 min | 5-10 min |

## 🔍 Model Insights

### Most Important Features
1. **Quantity Information** (pack sizes, weights)
2. **Brand Presence** (premium brands vs. generic)
3. **Product Category** (electronics, food, clothing)
4. **Text Complexity** (word count, description length)
5. **Quality Indicators** (luxury keywords, organic)

### Category Performance
- **Electronics**: Highest accuracy (clear specifications)
- **Food**: Good performance (quantity-driven pricing)
- **Clothing**: Moderate accuracy (brand-dependent)
- **Home/Beauty**: Variable performance

## 🛠️ Technical Details

### Dependencies
```bash
# Minimal (baseline)
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tqdm>=4.62.0
joblib>=1.0.0

# Full (enhanced/full)
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
requests>=2.25.0
nltk>=3.6.0
```

### Hardware Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 8GB+ for enhanced/full solutions
- **Storage**: 2GB for models and cache
- **GPU**: Optional (not required for baseline)

## 📝 Output Format

The solution generates `dataset/test_out.csv` with the required format:
```csv
sample_id,price
100179,20.61
245611,22.70
36806,32.38
...
```

## 🔒 Compliance

✅ **Model Constraints**: All models < 8B parameters   
✅ **No External Data**: Uses only provided training data  
✅ **No Price Lookup**: Pure ML approach  
