# Spectral HybridNet: EMG Gesture Recognition & Biometric Identification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A PyTorch-based Multi-Task Deep Learning Architecture for Real-Time Biosignal Classification**

---

## Overview

This repository contains my final-year engineering research project: a novel deep learning pipeline that simultaneously classifies hand gestures and identifies users from 8-channel surface EMG signals. 

The system tackles the **"Biometric Bottleneck"** – the inability of models to generalise across different users – by using a hybrid CNN-BiLSTM architecture with channel attention and multi-task learning.

**Key Achievements:**
- ✅ **98.3%** Gesture Classification Accuracy
- ✅ **95.3%** Biometric Identification Accuracy  
- ✅ **+5.43%** Improvement via Data Augmentation & Hyperparameter Tuning

---

## Model Architecture

The **Spectral HybridNet** combines three powerful paradigms:

1. **Spectral Feature Extraction**  
   - Short-Time Fourier Transform (STFT) converts raw EMG signals (8 channels) into 2D time-frequency spectrograms.

2. **Convolutional Backbone with Attention**  
   - A CNN with **Squeeze-and-Excitation (SE) blocks** learns spatial features and automatically re-weights the importance of each electrode channel.

3. **Bidirectional Temporal Modelling**  
   - A Bi-LSTM captures the evolution of muscle activation patterns across time.

4. **Multi-Task Learning**  
   - Dual classification heads predict *Gesture* (what the user is doing) and *Identity* (who the user is) simultaneously, forcing the network to learn richer, generalisable representations.

---

## Results

| Model | Gesture Accuracy | Biometric Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| Random Forest (Baseline) | 51.79% (LOSO) | - | Engineered Features |
| MaxCNN (CNN) | 65.89% | 52.57% | Raw EMG input |
| Bi-LSTM | 65.65% | - | Temporal modelling |
| TurboTransformer | 75.70% | - | Self-Attention |
| **Spectral HybridNet (Ours)** | **98.3% (Train) / 72.0% (Val)** | **95.3% (Train) / 79.0% (Val)** | **STFT + SE + BiLSTM + MTL** |
| MaxCNN (Tuned + Augmented) | - | **58.00%** | Noise Injection + GridSearch |

---

##  Tech Stack

- **Deep Learning**: PyTorch, TorchVision
- **Signal Processing**: SciPy (STFT, Filtering), NumPy, Pandas
- **ML Ops**: Scikit-learn (GridSearchCV, LOSO), Matplotlib (Visualisation)
- **Embedded Ready**: Exploring ONNX export and TensorFlow Lite for edge deployment.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Tmkaratigwa-AI/emg-gesture-recognition.git
cd emg-gesture-recognition
