
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from .mlp import MLP
import torch.nn.functional as F

class LogisticHead(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        # self.linear = nn.Linear(input_dim, 1)
        self.linear = MLP(
            input_size=input_dim,
            num_classes=1,
            hidden_sizes=[8,16,8,4],
            activation='gelu',
            dropout=0.1
        )

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class EnsembleModel(nn.Module):
    def __init__(self, mlp: nn.Module, vae: nn.Module):
        super().__init__()
        self.mlp = mlp
        self.vae = vae
        self.classifier = LogisticHead(input_dim=3)

    @staticmethod
    def _normalize(tensor: torch.Tensor):
        """Normalize tensor to [0, 1] range per batch."""
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    @torch.no_grad()
    def _reconstruction_loss(self, x):
        """Compute reconstruction MSE per sample."""
        mu, _ = self.vae.encode(x)
        recon = self.vae.decode(mu)
        return torch.mean((x - recon) ** 2, dim=1), mu

    @torch.no_grad()
    def _probs(self, x):
        logits = self.mlp(x)
        return F.softmax(logits, dim=-1)[:, 1]  # pathology probability

    def forward(self, x: torch.Tensor):
        rec_loss, mu = self._reconstruction_loss(x)
        # rec_loss = self._normalize(rec_loss)
        mlp_probs = self._probs(x)
        # features = torch.cat([mlp_probs.view(-1, 1), rec_loss.view(-1, 1), mu.flatten(start_dim=1)], dim=1) 
        features = torch.stack([mlp_probs, rec_loss, mu.flatten(start_dim=1).mean(dim=1)], dim=1) 
        out = self.classifier(features)
        return out  # shape [B]