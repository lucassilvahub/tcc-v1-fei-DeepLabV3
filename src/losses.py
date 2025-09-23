import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha = (
            self.alpha if self.alpha is not None else 1.0
        )  # <- garante que não seja None
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
