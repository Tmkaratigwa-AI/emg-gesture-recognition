%%writefile README.md
# EMG Signal Processing and Classification Pipeline

## Project Motivation
This project provides a robust pipeline for Electromyography (EMG) signal processing, feature extraction, and classification using various machine learning and deep learning models. The goal is to classify gestures and identify subjects based on EMG signals.

## Architecture
The codebase is modular:
- `config.py`: Central parameters.
- `src/utils/`: Signal processing and augmentation utilities.
- `src/models/`: Deep learning architectures (CNN, RNN, Transformer, Hybrid).
- `src/data/`: Data loading and dataset class.
- `src/training/`: Training loops and sklearn-compatible wrappers.
- `main.py`: Orchestrates the entire pipeline.

## Dataset
Uses the UCI EMG Data for Gestures dataset (`emg+data+for+gestures.zip`). It includes 8-channel EMG signals from multiple subjects performing gestures (cylindrical and tip).

## Results
Example accuracies (may vary):
- MaxCNN Gesture Validation: ~65.89%
- Bi-LSTM Gesture Validation: ~65.65%
- Transformer Gesture Validation: ~75.70%
- SpectralHybridNet Gesture Validation: ~72.00%
- SpectralHybridNet Biometric Validation: ~79.00%
- GridSearchCV Test Accuracy (Biometric): ~0.58
- Average LOSO Accuracy (RF): ~51.79%

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
