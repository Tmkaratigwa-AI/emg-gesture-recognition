import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

import config
from src.models.cnn import MaxCNN
from src.models.hybrid import SpectralHybridNet
from src.utils.augmentation import inject_noise


def train_spectral_hybrid(
    model: SpectralHybridNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = config.EPOCHS,
    device: torch.device = torch.device(config.DEVICE)
) -> Dict[str, List[float]]:
    """Train SpectralHybridNet with multi-task loss."""
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.ONE_CYCLE_MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    model.to(device)
    print(f"Training Spectral Hybrid on {device}...")

    history = {
        'train_loss': [], 'train_g_acc': [], 'train_s_acc': [],
        'val_loss': [], 'val_g_acc': [], 'val_s_acc': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss, train_g_acc, train_s_acc = 0, 0, 0

        for signals, sub_labels, gest_labels in train_loader:
            signals, sub_labels, gest_labels = signals.to(device), sub_labels.to(device), gest_labels.to(device)

            optimizer.zero_grad()
            g_out, s_out = model(signals)
            loss_g = criterion(g_out, gest_labels)
            loss_s = criterion(s_out, sub_labels)
            total_loss = config.MTL_GESTURE_LOSS_WEIGHT * loss_g + config.MTL_SUBJECT_LOSS_WEIGHT * loss_s
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_MAX_NORM)
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()
            train_g_acc += (g_out.argmax(1) == gest_labels).float().mean().item()
            train_s_acc += (s_out.argmax(1) == sub_labels).float().mean().item()

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_g_acc'].append(train_g_acc / len(train_loader))
        history['train_s_acc'].append(train_s_acc / len(train_loader))

        # Validation
        model.eval()
        val_loss, val_g_acc, val_s_acc = 0, 0, 0
        with torch.no_grad():
            for signals, sub_labels, gest_labels in val_loader:
                signals, sub_labels, gest_labels = signals.to(device), sub_labels.to(device), gest_labels.to(device)
                g_out, s_out = model(signals)
                loss_g = criterion(g_out, gest_labels)
                loss_s = criterion(s_out, sub_labels)
                total_loss = config.MTL_GESTURE_LOSS_WEIGHT * loss_g + config.MTL_SUBJECT_LOSS_WEIGHT * loss_s
                val_loss += total_loss.item()
                val_g_acc += (g_out.argmax(1) == gest_labels).float().mean().item()
                val_s_acc += (s_out.argmax(1) == sub_labels).float().mean().item()

        history['val_loss'].append(val_loss / len(val_loader))
        history['val_g_acc'].append(val_g_acc / len(val_loader))
        history['val_s_acc'].append(val_s_acc / len(val_loader))

        if config.VERBOSITY_LEVEL > 1 and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {history['train_loss'][-1]:.3f} | "
                  f"GAcc: {history['train_g_acc'][-1]*100:.1f}% | "
                  f"SAcc: {history['train_s_acc'][-1]*100:.1f}% | "
                  f"Val Loss: {history['val_loss'][-1]:.3f} | "
                  f"Val GAcc: {history['val_g_acc'][-1]*100:.1f}% | "
                  f"Val SAcc: {history['val_s_acc'][-1]*100:.1f}%")

    print("Training complete.")
    return history


class SklearnCompatibleMaxCNN(BaseEstimator):
    """Scikit-learn wrapper for MaxCNN with optional augmentation."""

    def __init__(
        self,
        num_classes: int = 2,
        lr: float = config.LEARNING_RATE,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        device: str = config.DEVICE,
        enable_augmentation: bool = config.ENABLE_AUGMENTATION,
        noise_std: float = config.NOISE_STD,
        random_state: Optional[int] = config.RANDOM_STATE
    ):
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self._device_obj = torch.device(device)
        self.enable_augmentation = enable_augmentation
        self.noise_std = noise_std
        self.random_state = random_state
        self.model = None
        self.criterion = None
        self.optimizer = None

    def _initialize_model(self):
        self.model = MaxCNN(num_classes=self.num_classes).to(self._device_obj)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SklearnCompatibleMaxCNN':
        self._initialize_model()

        X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 8, config.WINDOW_SIZE)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for signals, labels in loader:
                signals, labels = signals.to(self._device_obj), labels.to(self._device_obj)

                if self.enable_augmentation:
                    signals_np = signals.cpu().numpy()
                    aug_signals = inject_noise(signals_np, self.noise_std, self.random_state)
                    signals = torch.tensor(aug_signals, dtype=torch.float32).to(self._device_obj)

                self.optimizer.zero_grad()
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 8, config.WINDOW_SIZE)
        self.model.eval()
        preds = []

        with torch.no_grad():
            dataset = torch.utils.data.TensorDataset(X_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for batch in loader:
                signals = batch[0].to(self._device_obj)
                outputs = self.model(signals)
                predicted = outputs.argmax(1)
                preds.extend(predicted.cpu().numpy())

        return np.array(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


def perform_loso_cross_validation(
    X_features: np.ndarray,
    y_labels: np.ndarray,
    groups: np.ndarray,
    model_params: Dict[str, Any] = config.RF_LOSO_PARAMS,
    random_state: int = config.RANDOM_STATE,
    verbosity: int = config.VERBOSITY_LEVEL
) -> Tuple[Dict[str, float], float]:
    """Leave-One-Subject-Out cross-validation with RandomForest."""
    logo = LeaveOneGroupOut()
    scaler = StandardScaler()
    subject_accuracies: Dict[str, float] = {}
    all_preds, all_true = [], []

    unique_groups = np.unique(groups)
    id_to_name = {i: f'Subject_{i}' for i in unique_groups}

    if verbosity > 0:
        print(f"LOSO with RandomForest on {len(unique_groups)} subjects.")

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_features, y_labels, groups)):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(random_state=random_state, **model_params)
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        acc = accuracy_score(y_test, preds)

        sub_id = groups[test_idx][0]
        sub_name = id_to_name.get(sub_id, str(sub_id))
        subject_accuracies[sub_name] = acc * 100
        all_preds.extend(preds)
        all_true.extend(y_test)

        if verbosity > 1:
            print(f"Fold {fold_idx+1} ({sub_name}): {acc*100:.2f}")

    mean_acc = np.mean(list(subject_accuracies.values()))

    if verbosity > 0:
        print("\n" + "=" * 45)
        print("LOSO PERFORMANCE REPORT")
        print("=" * 45)
        for name, acc in subject_accuracies.items():
            print(f"{name}: {acc:.2f}")
        print(f"\nAverage LOSO Accuracy: {mean_acc:.2f}")
        print("Classification Report:")
        print(classification_report(all_true, all_preds))

    return subject_accuracies, mean_acc
