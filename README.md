# ⚽ Football Match Outcome Predictor

A deep learning project that predicts football match outcomes (Home Win / Draw / Away Win) using historical match data and betting odds. Built with PyTorch, this neural network classifier achieves robust predictions by learning from multiple seasons of football data.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training Details](#training-details)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a multi-class classification neural network to predict football match outcomes. The model takes 20 input features (including team statistics and betting odds) and predicts one of three outcomes:
- **Draw (D)** - Label 0
- **Home Win (H)** - Label 1
- **Away Win (A)** - Label 2

The model uses historical data from multiple seasons, with 2024 season data reserved for validation.

## ✨ Features

- **Custom PyTorch Dataset**: Implements `FootballDataset` class with automatic feature normalization
- **Deep Neural Network**: 3-layer fully connected network with dropout regularization
- **Class Imbalance Handling**: Uses weighted CrossEntropyLoss to handle imbalanced classes
- **Learning Rate Scheduling**: StepLR scheduler for adaptive learning
- **Comprehensive Training Loop**: Tracks both training and validation metrics
- **Visualization**: Generates loss and accuracy plots for model performance analysis
- **Model Persistence**: Saves best model weights for future inference

## 📁 Project Structure

```
Football Machine/
├── data/
│   ├── raw/                    # Raw CSV files from football-data.co.uk
│   └── processed/              # Processed and cleaned data
│       └── processed_df.csv    # Final processed dataset
├── src/
│   ├── data_prep.py           # Data loading and preprocessing utilities
│   ├── dataset.py             # PyTorch Dataset and DataLoader implementation
│   ├── model.py               # Neural network architecture
│   ├── train.py               # Training, validation, and testing functions
│   └── features.py            # Feature engineering (if applicable)
├── notebooks/                  # Jupyter notebooks for EDA and experimentation
│   ├── check.py
│   └── data_loader_notebook.py
├── tests/                      # Unit tests
│   ├── Test1.py
│   ├── test2.py
│   ├── test3.py
│   └── test4.py
├── main.py                     # Main training script
├── model.pth                   # Saved model weights
├── model_performance.png       # Training/validation metrics visualization
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd "Football Machine"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Place raw CSV files in `data/raw/`
   - Run data preparation (if needed):
     ```python
     from src.data_prep import combine_df, clean_df, save_df
     raw_df = combine_df()
     clean_data = clean_df(raw_df)
     save_df(clean_data)
     ```

## 💻 Usage

### Training the Model

Run the main training script:

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Create train/validation splits (pre-2024 for training, 2024 for validation)
3. Train the model for 10 epochs
4. Save the best model to `model.pth`
5. Generate performance plots in `model_performance.png`

### Using the Trained Model

```python
import torch
from src.model import FootballPredictor

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FootballPredictor(n_inputs=20).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(your_input_features)
    predicted_class = torch.argmax(predictions, dim=1)
```

## 🧠 Model Architecture

The `FootballPredictor` is a feedforward neural network with the following architecture:

```
Input Layer (20 features)
    ↓
Linear(20 → 32) + ReLU + Dropout(0.4)
    ↓
Linear(32 → 16) + ReLU + Dropout(0.5)
    ↓
Linear(16 → 3)
    ↓
Output (3 classes: Draw, Home Win, Away Win)
```

**Key Components:**
- **Activation Function**: ReLU for non-linearity
- **Regularization**: Dropout layers (0.4 and 0.5) to prevent overfitting
- **Output**: 3 logits for multi-class classification

## 📊 Dataset

### Input Features (20 total)
The model uses features derived from:
- Historical match statistics
- Betting odds (AvgH, AvgD, AvgA - average odds for Home, Draw, Away)
- Team performance metrics

### Data Split
- **Training**: All seasons except 2024
- **Validation**: 2024 season only

### Preprocessing
- Feature normalization using mean and standard deviation from training set
- Label encoding: D→0, H→1, A→2
- Missing values removed during cleaning

## 🎓 Training Details

### Hyperparameters
- **Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4 (L2 regularization)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss with class weights [1.2157, 0.7619, 1.1561]
- **LR Scheduler**: StepLR (step_size=7, gamma=0.1)

### Training Features
- **GPU Support**: Automatically uses CUDA if available
- **Reproducibility**: Fixed random seed (42)
- **Early Stopping**: Tracks minimum validation loss
- **Metrics Tracking**: Loss and accuracy for both train and validation sets

## 📈 Results

### Performance Metrics
- **Validation Accuracy**: ~48%
- **Model Selection**: Based on minimum validation loss (principled checkpoint selection, not best-of-N)

### Key Improvements
The model achieved this performance through several critical improvements:

1. **Betting Odds Features**: Incorporated average betting odds (AvgH, AvgD, AvgA) as key predictive features, which capture market expectations and provide strong signals for match outcomes.

2. **Fixed Dropout Bug**: Corrected dropout implementation to properly apply regularization during training while disabling it during validation/testing, preventing underfitting.

3. **Fixed Scheduler Decay**: Properly configured the StepLR scheduler (step_size=7, gamma=0.1) to decay learning rate at appropriate intervals, enabling better convergence.

4. **Principled Checkpoint Selection**: Model checkpoint is saved based on minimum validation loss rather than cherry-picking best accuracy, ensuring robust generalization and avoiding overfitting to validation set.

### Visualization
After training, the model generates `model_performance.png` with two plots:
1. **Training Loss vs Validation Loss**: Monitor overfitting and convergence
2. **Training Accuracy vs Validation Accuracy**: Track model performance across epochs

The best model (lowest validation loss) is automatically saved to `model.pth`.

### Context
Predicting football match outcomes is inherently challenging due to the high variance and unpredictability of sports. A ~48% accuracy on a 3-class problem (Draw/Home/Away) represents meaningful performance above random baseline (33.3%) and demonstrates the model's ability to learn patterns from historical data and betting market information.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more sophisticated feature engineering
- Implement cross-validation
- Experiment with different architectures (LSTM, Transformer)
- Add prediction confidence intervals
- Create a web interface for predictions
- Add more comprehensive testing

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Data source: [football-data.co.uk](http://www.football-data.co.uk/)
- Built with PyTorch, NumPy, Pandas, and Matplotlib

---

**Note**: This model is for educational purposes. Sports betting involves risk, and past performance does not guarantee future results.
