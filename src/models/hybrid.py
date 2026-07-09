import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channel: int, reduction: int = 4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpectralHybridNet(nn.Module):
    """Hybrid CNN+Attention+Bi-LSTM for multi-task learning."""

    def __init__(self, num_gestures: int = 2, num_subjects: int = 5, n_fft: int = 64, hop_length: int = 32):
        super(SpectralHybridNet, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.conv_block = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SELayer(64),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.lstm = nn.LSTM(input_size=64 * 8, hidden_size=128, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)

        self.fc_shared = nn.Linear(128 * 2, 128)
        self.gesture_head = nn.Linear(128, num_gestures)
        self.subject_head = nn.Linear(128, num_subjects)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, length = x.shape

        # STFT
        x_stft = torch.stft(
            x.view(-1, length),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            return_complex=True,
            window=torch.hann_window(self.n_fft).to(x.device)
        )
        x_spec = torch.abs(x_stft)
        x = x_spec.view(batch_size, channels, x_spec.shape[1], x_spec.shape[2])

        x = self.conv_block(x)  # (B, 64, 8, 8)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, 8, -1)  # (B, 8, 64*8)

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last time step

        x = self.dropout(F.relu(self.fc_shared(x)))
        return self.gesture_head(x), self.subject_head(x)


class GestureRNN(nn.Module):
    """Bi-LSTM for gesture classification."""

    def __init__(self, input_size: int = 8, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 2):
        super(GestureRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, C)
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)  # global average pooling
        out = self.dropout(out)
        return self.fc(out)


class TurboTransformer(nn.Module):
    """Optimized Transformer encoder."""

    def __init__(self, num_classes: int = 2, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super(TurboTransformer, self).__init__()
        self.input_proj = nn.Linear(8, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # (B, d_model)
        return self.fc(x)
