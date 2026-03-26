"""Unified model definitions for Bluetooth AoA indoor localization.

Three architectures are provided:
  * MLPRegressor      — flat-feature baseline
  * CNNRegressor      — 1-D convolutions over spatial spectra
  * TransformerRegressor — self-attention with a meta-token mechanism
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Learnable positional embedding added to token sequences."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos[:, : x.size(1), :]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, flat_feats, spec_seq, meta):
        return self.net(flat_feats)


class CNNRegressor(nn.Module):
    def __init__(self, meta_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        conv_out = 64 * 16
        self.head = nn.Sequential(
            nn.Linear(conv_out + meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, flat_feats, spec_seq, meta):
        x = spec_seq.permute(0, 2, 1)  # (B, 3, 324)
        x = self.conv(x)
        x = x.flatten(1)
        x = torch.cat([x, meta], dim=1)
        return self.head(x)


class TransformerRegressor(nn.Module):
    """Transformer encoder with a prepended meta token.

    The meta token (projected from gateway positions + timestamp) is
    concatenated with 324 spatial-spectrum tokens.  After encoding, the
    meta-token representation is extracted and fed through a regression
    head to predict the 2-D position.

    Architecture (default):
        d_model = 48, nhead = 6, num_layers = 2, dim_feedforward = 192
    """

    def __init__(
        self,
        meta_dim: int,
        d_model: int = 48,
        nhead: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        self.meta_proj = nn.Linear(meta_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, flat_feats, spec_seq, meta):
        spec_tokens = self.input_proj(spec_seq)          # (B, 324, d_model)
        meta_token = self.meta_proj(meta).unsqueeze(1)   # (B, 1, d_model)
        x = torch.cat([meta_token, spec_tokens], dim=1)  # (B, 325, d_model)
        x = self.pos(x)
        x = self.encoder(x)
        pooled = x[:, 0, :]  # meta-token representation
        x = torch.cat([pooled, meta], dim=1)
        return self.head(x)


def build_model(kind: str, flat_dim: int, meta_dim: int) -> nn.Module:
    """Factory function for creating a model by name."""
    kind = kind.lower()
    if kind == "mlp":
        return MLPRegressor(flat_dim)
    if kind == "cnn":
        return CNNRegressor(meta_dim=meta_dim)
    if kind == "transformer":
        return TransformerRegressor(meta_dim=meta_dim)
    raise ValueError(f"Unknown model kind: {kind}")
