"""PyTorch LSTM model for sequence-based loan default risk prediction."""

from __future__ import annotations

import torch
from torch import nn


class LSTMDefaultRiskModel(nn.Module):
    """Predict default probability from financial time-series sequences."""

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, sequence_batch: torch.Tensor) -> torch.Tensor:
        """Return default probability in [0, 1] for each input sequence."""

        lstm_out, _ = self.lstm(sequence_batch)
        last_hidden = lstm_out[:, -1, :]
        logits = self.head(last_hidden).squeeze(-1)
        return torch.sigmoid(logits)
