from salt.models import MaskFormerLossIman


class GLS(MaskFormerLossIman):
    """Geometry Loss Scaling (GLS)."""

    def __init__(self):
        super().__init__()

    def weight_loss(self, losses: dict):
        """Apply the loss weights to the loss dict."""
        return losses
