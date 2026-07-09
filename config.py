%%writefile config.py

import os

# --- 1. Global Settings ---
RANDOM_STATE = 42
VERBOSITY_LEVEL = 2
DEVICE = 'cuda'  # or 'cpu'

# --- 2. Data Settings ---
ZIP_FILE_NAME = 'emg+data+for+gestures.zip'
BASE_DATA_DIR = './emg_raw_data'
NESTED_ARCHIVE_FOLDER = 'EMG_data_for_gestures-master'
DATA_ROOT = os.path.join(BASE_DATA_DIR, NESTED_ARCHIVE_FOLDER)

ELITE_TARGETS = ['28', '26', '33', '34', '31']
GESTURE_IDS_TO_PROCESS = ['1', '2']

# --- 3. Signal Processing & Windowing ---
SAMPLING_RATE_HZ = 1000
BANDPASS_LOW_HZ = 20
BANDPASS_HIGH_HZ = 450
NOTCH_FILTER_FREQ_HZ = 50
NOTCH_FILTER_Q = 30

ZC_THRESHOLD = 0.01

WINDOW_SIZE = 250
OVERLAP = 125
CENTER_SLICE_PROPORTION = 0.50

# --- 4. Model Hyperparameters (Deep Learning) ---
CNN_PARAMS = {
    'num_classes': 2,
    'in_channels': 8,
    'conv1_out_channels': 32,
    'conv2_out_channels': 64,
    'kernel_size': 5,
    'stride': 1,
    'padding': 2,
    'pool_kernel_size': 2,
    'pool_stride': 2,
    'fc1_out_features': 128,
    'dropout_rate': 0.5
}

RNN_PARAMS = {
    'input_size': 8,
    'hidden_size': 64,
    'num_layers': 1,
    'num_classes': 2,
    'dropout': 0.3
}

TRANSFORMER_PARAMS = {
    'num_classes': 2,
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 128,
    'dropout': 0.1
}

HYBRID_NET_PARAMS = {
    'num_gestures': 2,
    'num_subjects': 5,
    'n_fft': 64,
    'hop_length': 32,
    'se_reduction_ratio': 4
}

# --- 5. Training Settings ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_MAX_NORM = 1.0

ONE_CYCLE_MAX_LR = 0.005

MTL_GESTURE_LOSS_WEIGHT = 1.5
MTL_SUBJECT_LOSS_WEIGHT = 0.5

GRID_SEARCH_BATCH_SIZE_OPTIONS = [16, 32, 64, 128]
GRID_SEARCH_EPOCHS_OPTIONS = [20, 30, 40, 50]
GRID_SEARCH_LR_OPTIONS = [0.0001, 0.0005, 0.001, 0.002]
GRID_SEARCH_CV_FOLDS = 3

# --- 6. Augmentation Settings ---
ENABLE_AUGMENTATION = True
NOISE_STD = 0.01
GRID_SEARCH_ENABLE_AUGMENTATION_OPTIONS = [True, False]
GRID_SEARCH_NOISE_STD_OPTIONS = [0.005, 0.01, 0.02]

# --- 7. Evaluation Settings ---
TEST_SET_SIZE = 0.20

RF_LOSO_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20
}
