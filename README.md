# Telugu Style Prompt Recovery for LLM

This repository contains the implementation for the **DravidianLangTech Prompt Recovery Task** focusing on Telugu style classification using transformer-based models.

## 📋 Project Overview

The goal of this project is to classify Telugu text into 9 different communicative styles:
- **Formal**
- **Informal**
- **Optimistic**
- **Pessimistic**
- **Humorous**
- **Serious**
- **Inspiring**
- **Authoritative**
- **Persuasive**

The model analyzes both the original transcript and the style-transformed version to identify the target communicative style.

## 🎯 Task Description

Given an original Telugu transcript and its style-transformed version, the model must predict which of the 9 communicative styles was applied to create the transformation.

**Input:** 
- Original Telugu text
- Style-transformed Telugu text

**Output:** 
- One of 9 style labels

## 🏗️ Model Architecture

- **Base Model:** `ai4bharat/IndicBERTv2-MLM-only`
- **Framework:** PyTorch (manual training loops, no Trainer API)
- **Custom Classification Head:** 
  - Dropout layer (p=0.1)
  - Linear classifier
  - Class-weighted CrossEntropyLoss for handling imbalanced data

## 📊 Data-Centric Approach

The implementation focuses on data quality and class balance prevention:

1. **Data Cleaning:**
   - Removal of empty/invalid texts
   - Fixed validation set (300 samples for reliable ground truth)
   - Duplicate detection and removal

2. **Class Collapse Prevention:**
   - Class-weighted loss function
   - Balanced sampling strategies
   - Early detection of class collapse during training
   - Curriculum learning with high-confidence samples first

3. **Curriculum Training Strategy:**
   - Stage 1: Train on high-confidence subset
   - Stage 2: Add medium-confidence samples (Tier 2)
   - Stage 3: Include all filtered data
   - Progressive complexity increase prevents collapse

## 🚀 Setup and Installation

### Requirements

```python
- Python 3.10+
- PyTorch (use Kaggle's preinstalled version on Kaggle)
- transformers
- scikit-learn
- sentencepiece
- tqdm
- pandas
- numpy
- matplotlib
- seaborn
```

### Installation

On Kaggle, run the package installation cell:

```python
import sys
!"{sys.executable}" -m pip install -q transformers scikit-learn sentencepiece tqdm
```

**Important:** Restart the kernel after installation!

### Local Setup

```bash
pip install torch transformers scikit-learn sentencepiece tqdm pandas numpy matplotlib seaborn
```

## 📁 Project Structure

```
Gitrepo/
├── telugu_style_prompt_recovery.ipynb  # Main Jupyter notebook with complete pipeline
├── submission.csv                       # Generated submission file
└── README.md                           # This file
```

## 💻 Usage

### On Kaggle

1. **Upload Data:**
   - Upload `PR_train.csv`, `PR_validation.csv`, and `PR_test.csv` to Kaggle
   - Adjust `DATA_DIR` in the first cell if needed

2. **Run the Notebook:**
   - Execute cells sequentially
   - Restart kernel after package installation
   - The notebook handles everything from data loading to submission generation

3. **Download Results:**
   - `submission.csv` - predictions for test set
   - Saved model (if model saving cell is executed)

### Locally

1. Place dataset files in the same directory as the notebook
2. Run all cells in sequence
3. Submission file will be generated automatically

## 🔑 Key Features

### 1. Manual PyTorch Training
- No Trainer API dependencies (Kaggle Python 3.12 compatible)
- Custom training loops with full control
- Explicit optimization and scheduling

### 2. Class Imbalance Handling
- Computed class weights from training distribution
- Weighted CrossEntropyLoss
- Stratified sampling for validation set

### 3. Robust Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrix visualization
- Per-class performance metrics
- Class collapse detection

### 4. Reproducibility
- Fixed random seeds (RANDOM_SEED = 42)
- Deterministic CUDA operations
- Controlled sampling strategies

## 📈 Training Configuration

```python
RANDOM_SEED = 42
MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
```

### Training Strategy

- **Optimizer:** AdamW
- **Scheduler:** Linear warmup (10% of total steps)
- **Gradient Clipping:** Max norm 1.0
- **Epochs:** 5 (Stage 1) + 3 (Stage 2) + additional epochs for full dataset

## 🎓 Curriculum Learning

The model uses a 3-stage curriculum:

1. **Stage 1 (Epochs 1-5):** High-confidence samples only
2. **Stage 2 (Epochs 6-8):** High + Medium confidence samples
3. **Stage 3 (Remaining):** All filtered data

This prevents class collapse and improves model stability.

## 📝 Output Format

Submission file format:
```csv
id,label
1,Formal
2,Humorous
3,Serious
...
```

## 🔍 Model Evaluation

The notebook includes comprehensive evaluation:

- Training/validation metrics per epoch
- Confusion matrix visualization
- Classification report with per-class metrics
- Class collapse detection warnings
- Submission file verification

## 🛠️ Troubleshooting

### Common Issues

1. **`NameError: EVAL_BATCH_SIZE not defined`**
   - Fixed: Uses `BATCH_SIZE` for all DataLoaders

2. **`AttributeError: 'StyleClassifier' has no attribute 'save_pretrained'`**
   - Fixed: Uses `torch.save(model.state_dict(), path)` for custom models

3. **Class Collapse (model predicts only 1-2 classes)**
   - Implemented class-weighted loss
   - Curriculum learning strategy
   - Early detection and warnings

4. **PyTreeSpec binary incompatibility**
   - Solution: Use Kaggle's preinstalled PyTorch, don't reinstall

## 📊 Performance Metrics

The model is evaluated using:
- **Macro F1-Score** (primary metric)
- Accuracy
- Precision
- Recall
- Per-class performance

## 🤝 Contributing

This is a competition submission. For questions or issues, please open an issue in the repository.

## 📄 License

This project is part of the DravidianLangTech shared task.

## 🙏 Acknowledgments

- **AI4Bharat** for the IndicBERTv2 model
- **DravidianLangTech** organizers for the shared task
- **Hugging Face** for the Transformers library

## 📚 References

- [IndicBERTv2](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only)
- [DravidianLangTech Workshop](https://sites.google.com/view/dravidianlangtech-2025/)

---

**Note:** This implementation is optimized for Kaggle's Python 3.12 environment with manual PyTorch training loops.
