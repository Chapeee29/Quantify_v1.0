from __future__ import annotations

import torch
from torch import nn


class DeepSequenceModel(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        static_dim: int,
        hidden_size: int = 96,
        num_layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=sequence_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static = nn.Sequential(
            nn.Linear(static_dim, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + max(16, hidden_size // 2), hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(sequence)
        encoded = hidden[-1]
        static_encoded = self.static(static)
        logits = self.head(torch.cat([encoded, static_encoded], dim=1))
        return logits.squeeze(1)
