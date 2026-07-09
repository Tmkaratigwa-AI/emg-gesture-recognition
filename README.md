# EMG Gesture Recognition & Biometric Identification System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

> **A production-ready PyTorch pipeline for real-time EMG gesture recognition and biometric identification using hybrid deep learning architectures.**

---

## 📖 Abstract

Surface electromyography (sEMG) has emerged as a transformative modality in human-computer interaction, enabling intuitive control of prosthetic devices, robotic systems, and virtual reality environments. However, practical deployment remains constrained by the **"Biometric Bottleneck"** – the inability of models to generalise across diverse users due to high inter-subject physiological variability.

This project presents **Spectral HybridNet**, a novel multi-task deep learning architecture that simultaneously performs **gesture recognition** (what the user is doing) and **biometric identification** (who the user is) from 8-channel sEMG data. By integrating **STFT spectral fingerprints**, **Squeeze-and-Excitation channel attention**, and **Bi-LSTM temporal modelling** within a unified framework, the system achieves robust, real-time classification that outperforms single-task baselines.

---

## 🎯 Key Features

| **Feature** | **Description** |
|:------------|:----------------|
| **Multi-Task Learning** | Simultaneous gesture and identity prediction – richer representations |
| **Spectral Front-End** | STFT transforms raw EMG into 2D time-frequency spectrograms |
| **Channel Attention** | Squeeze-Excitation blocks re-weight electrode importance |
| **Temporal Modelling** | Bi-LSTM captures muscle activation sequences over time |
| **Robust Validation** | LOSO cross-validation, GridSearchCV, data augmentation |
| **Modular Design** | Clean, extensible codebase with scikit-learn compatible wrappers |

---

## 🧠 Model Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   RAW EMG INPUT                    │
                    │             8 channels × 250 samples               │
                    └─────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────────┐
                    │         SHORT-TIME FOURIER TRANSFORM (STFT)        │
                    │     Time-Frequency Spectrograms (8 × 33 × 8)       │
                    └─────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────────┐
                    │              CNN BACKBONE + ATTENTION              │
                    │  Conv2d → BN → ReLU → MaxPool → SE Block          │
                    └─────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────────┐
                    │              BIDIRECTIONAL LSTM                    │
                    │     2-layer Bi-LSTM with 128 hidden units          │
                    └─────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────────┐
                    │             MULTI-TASK HEADS                       │
                    ├─────────────────────┬───────────────────────────────┤
                    │   GESTURE HEAD      │   BIOMETRIC HEAD              │
                    │  (Cylindrical/Tip)  │  (Subject 1-5)                │
                    └─────────────────────┴───────────────────────────────┘
```

---

## 📊 Results

| **Model** | **Task** | **Accuracy** | **Notes** |
|:----------|:---------|:-------------|:----------|
| Random Forest (LOSO) | Gesture | **51.79%** | Engineered features, cross-subject |
| MaxCNN | Gesture | **65.89%** | 1D CNN on raw EMG |
| Bi-LSTM | Gesture | **65.65%** | Bidirectional RNN |
| TurboTransformer | Gesture | **75.70%** | Self-attention baseline |
| **Spectral HybridNet (Val)** | Gesture | **72.00%** | **STFT + SE + BiLSTM + MTL** |
| **Spectral HybridNet (Val)** | Biometric | **79.00%** | Subject identification |
| MaxCNN (Tuned + Augmented) | Biometric | **58.00%** | Noise injection + GridSearch |

---

## 🏗️ Project Structure

```
emg-gesture-recognition/
│
├── main.py                   # Master training & evaluation script
├── config.py                 # Centralised configuration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py    # EMG data loading, preprocessing, windowing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py            # MaxCNN architecture
│   │   └── hybrid.py         # SpectralHybridNet, Bi-LSTM, Transformer
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py        # Training loops, LOSO, sklearn wrappers
│   │
│   └── utils/
│       ├── __init__.py
│       ├── signal_processing.py  # Filtering, STFT, feature extraction
│       └── augmentation.py       # Noise injection, amplitude scaling
│
└── docs/
    └── dissertation_abstract.md  # Research abstract
```

---

## 📦 Dataset

This project uses the **UCI EMG Data for Gestures** dataset.

**Key characteristics:**
- **Sensors:** 8-channel MYO armband
- **Sampling Rate:** 1000 Hz
- **Subjects:** 5 "elite" subjects (28, 26, 33, 34, 31)
- **Gestures:** Cylindrical grasp (1) and Tip pinch (2)
- **Format:** Space-separated `.txt` files

**Download:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Tmkaratigwa-AI/emg-gesture-recognition.git
cd emg-gesture-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download and Extract Dataset

1. Download `emg+data+for+gestures.zip` from the UCI repository
2. Place it in the project root directory
3. The pipeline will automatically extract it to `./emg_raw_data/`

### 4. Run the Pipeline

```bash
python main.py
```

This will:
- ✅ Load and preprocess the data
- ✅ Train all models (MaxCNN, Bi-LSTM, Transformer, SpectralHybridNet)
- ✅ Run GridSearchCV for hyperparameter optimisation
- ✅ Perform LOSO cross-validation
- ✅ Generate visualisation plots (`pipeline_results.png`)

---

## ⚙️ Configuration

All hyperparameters and paths are centralised in `config.py`:

```python
# Data paths
DATA_DIR = './emg_raw_data'
ELITE_TARGETS = ['28', '26', '33', '34', '31']

# Signal processing
SAMPLING_RATE_HZ = 1000
WINDOW_SIZE_MS = 250
OVERLAP_MS = 125

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = 'cuda'  # or 'cpu'

# Model-specific
HYBRID_NET_PARAMS = {
    'n_fft': 64,
    'hop_length': 32,
}
```

Modify `config.py` to adjust parameters or use different subjects.

---

## 📊 Outputs

The pipeline generates:

1. **Training Curves** – Loss, Gesture Accuracy, Subject Accuracy
2. **Confusion Matrix** – Gesture classification performance
3. **LOSO Bar Chart** – Per-subject cross-validation accuracy
4. **Console Summary** – All model accuracies

All plots are saved as `pipeline_results.png`.

---

## 📖 Research Context

This project was completed as part of my Bachelor's degree in Electronic Engineering. The dissertation explores the difficulties of inter-subject generalisation in EMG interfaces and proposes the **SpectralHybridNet** as a robust solution.

> *"EMG Gesture Recognition System" – Full dissertation available upon request.*

[View Dissertation Abstract](docs/dissertation_abstract.md)

---

## 🛠️ Tech Stack

| **Category** | **Technologies** |
|:-------------|:-----------------|
| **Deep Learning** | PyTorch, TorchVision |
| **Signal Processing** | SciPy (STFT, Filtering), NumPy, Pandas |
| **ML Ops** | Scikit-learn (GridSearchCV, LOSO), Matplotlib, Seaborn |
| **Deployment** | ONNX (explored), TensorFlow Lite (planned) |
| **Version Control** | Git, GitHub |

---

## 🤝 Connect with Me

- **GitHub:** [github.com/Tmkaratigwa-AI](https://github.com/Tmkaratigwa-AI)
- **LinkedIn:** [linkedin.com/in/tatenda-mkaratigwa](linkedin.com/in/tatenda-mkaratigwa-bb290626a-)
-  **Email:** tmkaay21@gmail.com
- **Portfolio:** [github.com/Tmkaratigwa-AI](https://github.com/Tmkaratigwa-AI)

---

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **UCI Machine Learning Repository** – for providing the EMG dataset
- **My academic supervisors** – for guidance on signal processing and deep learning
- **The open-source community** – for PyTorch, scikit-learn, and the tools that made this possible

---

## ⭐ Support

If you find this project useful, please consider giving it a star ⭐ on GitHub!

---

## 🏆 What Makes This Project Stand Out

| **Aspect** | **Why It Matters** |
|:-----------|:-------------------|
| **Multi-Task Learning** | Simultaneous gesture + identity prediction forces richer representations |
| **Spectral Front-End** | STFT spectrograms are more robust than raw EMG or time-domain features |
| **Channel Attention** | SE blocks automatically weight important electrodes – no manual selection |
| **LOSO Validation** | Gold-standard for cross-subject generalisation – proves real-world viability |
| **Data Augmentation** | Noise injection improved accuracy by **+5.43%** – shows engineering rigour |
| **Modular Codebase** | Clean, extensible, scikit-learn compatible – production-ready |

---

**Built with ❤️ by Tatenda Mkaratigwa**
