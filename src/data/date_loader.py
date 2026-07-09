import os
import glob
import zipfile
import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional

import config


def load_and_process_emg_data(
    zip_file_name: str = config.ZIP_FILE_NAME,
    base_data_dir: str = config.BASE_DATA_DIR,
    nested_archive_folder: str = config.NESTED_ARCHIVE_FOLDER,
    elite_targets: List[str] = config.ELITE_TARGETS,
    gesture_ids_to_process: List[str] = config.GESTURE_IDS_TO_PROCESS,
    window_size: int = config.WINDOW_SIZE,
    overlap: int = config.OVERLAP,
    center_slice_proportion: float = config.CENTER_SLICE_PROPORTION
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Load, process, and window EMG data from a zip file."""
    X_raw_list, y_labels, subject_groups = [], [], []

    zip_file_path = os.path.join('/content', zip_file_name)
    data_root = os.path.join(base_data_dir, nested_archive_folder)

    # Extraction logic
    if not os.path.exists(data_root):
        if os.path.exists(base_data_dir) and not os.path.exists(os.path.join(base_data_dir, nested_archive_folder)):
            data_root = base_data_dir
        if not os.path.exists(data_root):
            if os.path.exists(zip_file_path):
                os.makedirs(base_data_dir, exist_ok=True)
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(base_data_dir)
                if not os.path.exists(data_root) and os.path.exists(os.path.join(base_data_dir, nested_archive_folder)):
                    data_root = os.path.join(base_data_dir, nested_archive_folder)
                else:
                    data_root = base_data_dir
            else:
                raise FileNotFoundError(f"Zip file or data directory not found.")

    if not os.path.exists(data_root) or not os.listdir(data_root):
        raise FileNotFoundError(f"Data root '{data_root}' is empty or missing.")

    all_txt_files = glob.glob(os.path.join(data_root, '**/*.txt'), recursive=True)
    if not all_txt_files:
        raise FileNotFoundError(f"No .txt files found in '{data_root}'.")

    subject_file_map: Dict[str, List[str]] = {}

    for f_path in all_txt_files:
        parts = f_path.split(os.sep)
        sub_id_found = None
        for part in parts:
            if part.isdigit() and len(part) <= 2:
                sub_id_found = part
                break
        if sub_id_found in elite_targets:
            subject_file_map.setdefault(sub_id_found, []).append(f_path)

    if not subject_file_map:
        raise ValueError(f"No data found for ELITE_TARGETS {elite_targets}.")

    unique_subjects = sorted(subject_file_map.keys())
    internal_subject_id_map = {name: i for i, name in enumerate(unique_subjects)}

    for sub_id_str, f_paths in subject_file_map.items():
        for f_path in f_paths:
            gesture_id_str = os.path.basename(f_path).split('_')[0]
            if gesture_id_str in gesture_ids_to_process:
                try:
                    raw_df = pd.read_csv(f_path, sep=r'\s+', engine='python', header=None)
                    raw_df = raw_df.apply(pd.to_numeric, errors='coerce').dropna()
                    if raw_df.empty:
                        continue
                    data_vals = raw_df.values
                    if data_vals.shape[1] >= 9:
                        emg_slice = data_vals[:, 1:9]
                    elif data_vals.shape[1] == 8:
                        emg_slice = data_vals[:, 0:8]
                    else:
                        continue

                    emg_normalized = zscore(emg_slice, axis=0)
                    total_len = len(emg_normalized)
                    mid = total_len // 2
                    offset = int(total_len * (center_slice_proportion / 2))
                    active_data = emg_normalized[mid - offset: mid + offset]

                    step = window_size - overlap
                    if step <= 0:
                        raise ValueError(f"Invalid window step: window_size={window_size}, overlap={overlap}")

                    for i in range(0, len(active_data) - window_size + 1, step):
                        win = active_data[i: i + window_size]
                        if len(win) == window_size:
                            X_raw_list.append(win)
                            y_labels.append(int(gesture_id_str))
                            subject_groups.append(internal_subject_id_map[sub_id_str])

                except Exception as e:
                    print(f"Error processing {f_path}: {e}")
                    continue

    if not X_raw_list:
        raise ValueError("No windows extracted. Check data and parameters.")

    return X_raw_list, np.array(y_labels), np.array(subject_groups)


class EMGDataset(Dataset):
    """PyTorch Dataset for EMG signals with subject and gesture labels."""

    def __init__(self, signals: List[np.ndarray], h_labels: np.ndarray, g_labels: np.ndarray):
        self.signals = torch.tensor(np.array(signals), dtype=torch.float32).permute(0, 2, 1)

        unique_h = np.unique(h_labels)
        h_map = {val: i for i, val in enumerate(unique_h)}
        self.h_labels = torch.tensor([h_map[val] for val in h_labels], dtype=torch.long)
        self.num_subjects = len(unique_h)

        unique_g = np.unique(g_labels)
        g_map = {val: i for i, val in enumerate(unique_g)}
        self.g_labels = torch.tensor([g_map[val] for val in g_labels], dtype=torch.long)
        self.num_gestures = len(unique_g)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.h_labels[idx], self.g_labels[idx]
