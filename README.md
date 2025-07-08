
# 🐜 Termite Detection using Pre-trained Deep Learning Models

This project implements a system for detecting termites based on **spectrogram images** using various **pre-trained deep learning models**. The R Markdown notebook `R-models.Rmd` orchestrates the Python environment setup, data loading, model training, and evaluation.

---

## 📚 Table of Contents
- [Project Overview](#project-overview)  
- [Architecture](#architecture)  
- [Setup and Installation](#setup-and-installation)  
- [Data Preparation](#data-preparation)  
- [Model Training and Evaluation](#model-training-and-evaluation)  
- [Results](#results)  
- [Usage](#usage)  
- [Contributing](#contributing)  

---

## 📌 Project Overview

The goal of this project is to classify spectrogram images as either containing **"Termite"** or **"No Termite"** activity. It leverages **transfer learning** using pre-trained models from the `timm` library in Python, accessed and managed via `reticulate` in R.

### 🔧 Key Features:
- Automated Python environment setup  
- Flexible data loading with dummy data creation for testing  
- Custom PyTorch Dataset for spectrograms  
- PretrainedTermiteClassifier using image classification backbones (EfficientNet, ResNet, ViT)  
- Training and evaluation pipeline with detailed metrics  

---

## 🧱 Architecture

### 🗃️ Data Management (R & Python)
- `check_data_files()`: Verifies existence of `train.csv`, `val.csv`, `test.csv`  
- `create_dummy_data()`: Generates synthetic `.rds` spectrograms for testing  
- `SpectrogramDataset`: Custom PyTorch `Dataset` to load `.npy` or `.rds` files and prepare them for model input  

### 🧠 Model Architecture (PyTorch)
- `PretrainedTermiteClassifier`:
  - Loads a pre-trained backbone via `timm.create_model()`
  - Appends a custom classifier head (Dropout → Linear → ReLU → Softmax)  
  - Supports models like `efficientnet_b0`, `resnet50`, `vit_base_patch16_224`

### 🔄 Training & Evaluation
- `create_transforms()`: Data augmentation for training; resizing/normalization for validation and testing  
- `create_dataloaders()`: Wraps datasets into PyTorch `DataLoader` objects  
- `train_model()`: Full training loop with:
  - CrossEntropyLoss, AdamW, LR scheduler
  - GPU support
  - Saves best model based on validation accuracy  
- `evaluate_model()`: Calculates:
  - Test loss, accuracy
  - Classification report
  - Confusion matrix  

---

## ⚙️ Setup and Installation

### ✅ Requirements:
- R and RStudio  
- Python (with `reticulate` access)  

### 🛠️ R Setup:
```r
install.packages(c("reticulate", "readr", "dplyr", "ggplot2", "purrr", "stringr"))
```

### 🛠️ Python Setup (via reticulate):
```r
library(reticulate)

# Recommended: create a virtual environment
# virtualenv_create("r-reticulate-env")
# use_virtualenv("r-reticulate-env", required = TRUE)

# Or use a specific Python interpreter
# use_python("/path/to/python", required = TRUE)

# Then run:
setup_python_environment()
```

`setup_python_environment()` installs:
- torch, torchvision, torchaudio  
- transformers, timm, librosa  
- scikit-learn, numpy, pandas  
- matplotlib, seaborn, pillow  
- opencv-python, soundfile  

---

## 📁 Data Preparation

### ✅ Directory Structure:
The system looks for CSV and spectrogram files in one of:
- `data/processed`
- `data`
- `processed`

### ✅ CSV Format:
Files required:
- `train.csv`
- `val.csv`
- `test.csv`

Each must have:
- `file_path`: path to spectrogram file (`.npy` or `.rds`)  
- `termite_flag`: `0` (no termite) or `1` (termite)

### 🧪 Dummy Data:
Use `create_dummy_data()` to auto-generate sample CSVs and `.rds` spectrograms for quick testing.

---

## 🏋️ Model Training and Evaluation

### 🔄 Run Full Pipeline via:
```r
run_termite_detection_safe(
  model_name = "efficientnet_b0",   # or resnet50, vit_base_patch16_224, etc.
  num_epochs = 50,
  batch_size = 8,
  learning_rate = 1e-4,
  create_dummy = FALSE              # Set TRUE for synthetic data
)
```

### 🔧 Parameters:
- `model_name`: name of the pre-trained backbone (from `timm`)  
- `num_epochs`: number of training epochs  
- `batch_size`: number of samples per batch  
- `learning_rate`: learning rate  
- `create_dummy`: generate dummy data if set to `TRUE`

---

## 📊 Results

Each run of `run_termite_detection_safe()` returns:
- `model`: the trained PyTorch model  
- `training_history`: list of train/val losses and accuracy  
- `test_results`: dictionary with:
  - predictions, labels  
  - accuracy, test loss  
  - confusion matrix  
  - classification report  

### 🔍 Example Output:
```r
start_resNet50 <- run_termite_detection_safe(
  model_name = "resnet50",
  num_epochs = 50,
  batch_size = 8,
  learning_rate = 1e-4,
  create_dummy = FALSE
)

start_resNet50$test_results
```

---

## ▶️ Usage

1. **Place your Data**:  
   Ensure `train.csv`, `val.csv`, `test.csv` and spectrograms (`.npy` or `.rds`) are in the correct directory (`data/processed` preferred).

2. **Open `R-models.Rmd` in RStudio**  

3. **Run Chunks Sequentially**:
   - Load libraries  
   - Run `check_data_files()` and optionally `create_dummy_data()`  
   - Run `setup_python_environment()` and `setup_python_modules()`  
   - Load datasets via `load_datasets()`  
   - Train models via `run_termite_detection_safe()` or one of the presets

---

## 📦 Model Shortcuts

```r
# EfficientNet-b0
start_efficientnet_b0 <- run_termite_detection_safe(
  model_name = "efficientnet_b0",
  num_epochs = 50,
  batch_size = 8,
  learning_rate = 1e-4,
  create_dummy = FALSE
)

# ResNet50
start_resNet50 <- run_termite_detection_safe(
  model_name = "resnet50",
  num_epochs = 50,
  batch_size = 8,
  learning_rate = 1e-4,
  create_dummy = FALSE
)

# ViT Base
start_vit_base_patch16_224 <- run_termite_detection_safe(
  model_name = "vit_base_patch16_224",
  num_epochs = 50,
  batch_size = 8,
  learning_rate = 1e-4,
  create_dummy = FALSE
)
```


