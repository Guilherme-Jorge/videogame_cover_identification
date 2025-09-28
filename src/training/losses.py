"""Loss functions for training."""

import torch
import torch.nn as nn


class NTXent(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for contrastive learning."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z1: Embeddings from first view.
            z2: Embeddings from second view.

        Returns:
            Loss value.
        """
        z = torch.cat([z1, z2], dim=0)
        sim = (z @ z.t()) / self.t
        b = z1.size(0)
        mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        targets = torch.arange(b, device=z.device)
        targets = torch.cat([targets + b, targets])
        return nn.functional.cross_entropy(sim, targets)
