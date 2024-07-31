import torch
import torch.nn.functional as F

from salt.models import MaskFormerLossIman


class RLW(MaskFormerLossIman):
    """Random Loss Weights (RLW)."""

    def __init__(self):
        super().__init__()

    def weight_loss(self, losses: dict):
        """Apply the loss weights to the loss dict."""
        task_num = len(losses)
        weights = F.softmax(torch.randn(task_num), dim=-1)

        # Apply weights to each loss
        for i, k in enumerate(list(losses.keys())):
            losses[k] *= weights[i]
        return losses
