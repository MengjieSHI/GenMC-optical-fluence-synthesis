"""Loss functions for training the GenMC conditional GAN.

The training objective (Methods, Eq. 1) combines an adversarial cGAN loss with an
L1 reconstruction loss::

    V(G, D) = min_G max_D  L_cGAN(G, D) + beta * L_L1(G)

with ``beta = 10`` by default. Discriminators in this code base output raw logits,
so :class:`~torch.nn.BCEWithLogitsLoss` is used for numerical stability.
"""

from __future__ import annotations

import torch
from torch import nn


class GeneratorLoss(nn.Module):
    """Generator objective: non-saturating adversarial loss + L1 reconstruction.

    Args:
        l1_weight: weight ``beta`` of the L1 reconstruction term (default 10).
        adv_weight: weight of the adversarial term (default 1).
        reconstruction: ``"l1"`` (paper default) or ``"l2"``.
    """

    def __init__(self, l1_weight: float = 10.0, adv_weight: float = 1.0, reconstruction: str = "l1"):
        super().__init__()
        self.l1_weight = l1_weight
        self.adv_weight = adv_weight
        self.bce = nn.BCEWithLogitsLoss()
        if reconstruction == "l1":
            self.recon = nn.L1Loss()
        elif reconstruction == "l2":
            self.recon = nn.MSELoss()
        else:
            raise ValueError("reconstruction must be 'l1' or 'l2'")

    def forward(self, fake: torch.Tensor, real: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        # The generator wants the discriminator to classify generated maps as real (label 1).
        real_label = torch.ones_like(fake_pred)
        adversarial = self.bce(fake_pred, real_label)
        reconstruction = self.recon(fake, real)
        return self.adv_weight * adversarial + self.l1_weight * reconstruction


class DiscriminatorLoss(nn.Module):
    """Discriminator objective: classify real maps as 1 and generated maps as 0."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, fake_pred: torch.Tensor, real_pred: torch.Tensor) -> torch.Tensor:
        fake_loss = self.bce(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.bce(real_pred, torch.ones_like(real_pred))
        return 0.5 * (fake_loss + real_loss)
