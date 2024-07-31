from salt.models import MaskFormerLossIman


class StaticWeighting(MaskFormerLossIman):
    """Uncertainty Weights (UW)."""

    def __init__(self):
        super().__init__()

    def weight_loss(self, losses: dict):
        """Apply the loss weights to the loss dict."""
        for k in list(losses.keys()):
            losses[k] *= self.loss_weights[k]
        return losses
