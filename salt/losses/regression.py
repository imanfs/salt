import torch
import torch.nn as nn


class RegressionLoss(nn.Module):
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        target_idx: int = 0,
        pt_idx: int = None,
        uncertainty=False,
    ):
        """Jet cross entropy classification loss.

        Parameters
        ----------
        weight : str
            Name of the loss, used to access relevant preds and labels.
        weight : float, optional
            Apply weighting to the computed loss, by default 1.0
        target_idx : int, optional
            The index of the label in the 'additional_labels' dataset that
            you wish to regress
        pt_incd
        """
        super().__init__()
        self.name = name
        self.weight = weight
        self.uncertainty = uncertainty
        if uncertainty:
            self.loss = self.uncert_loss
        else:
            self.loss = nn.MSELoss()
        self.target_idx = target_idx
        self.pt_idx = pt_idx

    def uncert_loss(self, preds, true):
        """https://towardsdatascience.com/get-uncertainty-estimates-in-neural-
        networks-for-free-48f2edb82c8f https://arxiv.org/abs/2204.09308."""
        # preds[0] = prediction
        # preds[1] = ln(std)
        std = torch.exp(preds[:, 1])
        # true.shape = [n,1] (for consistency to work with other loss)
        t1 = 2 * preds[:, 1] + torch.square((preds[:, 0] - true[:, 0]) / std)
        n = len(preds)
        # print(t1.shape)
        res = 1 / n * torch.sum(t1)
        # print(res.shape, n)
        return res

    def forward(self, preds, true):
        # Get the idx of all index with non-zero target
        # (we assume 0 is to be ignored)
        non_zero_idx = true[self.name][:, self.target_idx] != 0

        preds = preds[self.name]
        true_target = true[self.name][:, self.target_idx]

        # Scale by pt if desired
        if self.pt_idx >= 0:
            pt = true[self.name][:, self.pt_idx]
            true_target = true_target / pt

        return (
            self.loss(preds[non_zero_idx], true_target[non_zero_idx, None])
            * self.weight
        )
