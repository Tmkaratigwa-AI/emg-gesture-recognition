# ==========================================
# CONFIGURATION
# ==========================================

# --- Paths ---
ZIP_FILE_NAME = 'emg+data+for+gestures.zip'
BASE_DATA_DIR = './emg_raw_data'
NESTED_ARCHIVE_FOLDER = 'EMG_data_for_gestures-master'
ELITE_TARGETS = ['28', '26', '33', '34', '31']
GESTURE_IDS_TO_PROCESS = ['1', '2']

# --- Data Processing ---
SAMPLING_RATE_HZ = 1000
WINDOW_SIZE_MS = 250
WINDOW_SIZE = int(SAMPLING_RATE_HZ * WINDOW_SIZE_MS / 1000)  # 250
OVERLAP_MS = 125
OVERLAP = int(SAMPLING_RATE_HZ * OVERLAP_MS / 1000)         # 125
CENTER_SLICE_PROPORTION = 0.5
ZC_THRESHOLD = 0.01

# --- Training ---
DEVICE = 'cuda'
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42
VERBOSITY_LEVEL = 1
GRADIENT_CLIP_MAX_NORM = 1.0

# --- Multi-Task Learning Weights ---
MTL_GESTURE_LOSS_WEIGHT = 1.5
MTL_SUBJECT_LOSS_WEIGHT = 0.5

# --- OneCycleLR ---
ONE_CYCLE_MAX_LR = 0.005

# --- CNN Params ---
CNN_PARAMS = {
    'in_channels': 8,
    'num_classes': 2,
}

# --- Hybrid Net Params ---
HYBRID_NET_PARAMS = {
    'n_fft': 64,
    'hop_length': 32,
}

# --- GridSearchCV Params ---
GRID_SEARCH_CV_FOLDS = 3
GRID_SEARCH_LR_OPTIONS = [0.0001, 0.0005, 0.001, 0.002]
GRID_SEARCH_EPOCHS_OPTIONS = [20, 30, 40, 50]
GRID_SEARCH_BATCH_SIZE_OPTIONS = [16, 32, 64, 128]
GRID_SEARCH_ENABLE_AUGMENTATION_OPTIONS = [True, False]
GRID_SEARCH_NOISE_STD_OPTIONS = [0.005, 0.01, 0.02]

# --- Random Forest LOSO Params ---
RF_LOSO_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': RANDOM_STATE,
}

# --- Augmentation ---
ENABLE_AUGMENTATION = True
NOISE_STD = 0.01
