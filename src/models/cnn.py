import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxCNN(nn.Module):
    """1D CNN for EMG gesture classification."""

    def __init__(self, num_classes: int = 2):
        super(MaxCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        # After two conv+pool: shape (B, 64, 62)
        self.fc1 = nn.Linear(64 * 62, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 32, 125) -> (B, 32, 62)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, 64, 62)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
