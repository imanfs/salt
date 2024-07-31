import torch
from torch import nn

from salt.models import MaskFormerLossIman


class UW(MaskFormerLossIman):
    """Uncertainty Weights (UW).

    This method is proposed in `Multi-Task Learning Using Uncertainty to Weigh Losses for
    Scene Geometry and Semantics (CVPR 2018) and implemented by us.
    """

    def __init__(self):
        super().__init__()

    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([-0.5] * self.task_num))

    def backward(self, losses):
        loss = (losses / (2 * self.loss_scale.exp()) + self.loss_scale / 2).sum()
        loss.backward()
        return (1 / (2 * torch.exp(self.loss_scale))).detach().cpu().numpy()
