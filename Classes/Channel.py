import numpy as np

class Channel:
    """
    Gaussian channel with optional packet loss.
    """

    def __init__(self, noise_std=0.5, packet_loss=0.0, rng=None):
        self.noise_std = noise_std
        self.packet_loss = packet_loss
        self.rng = rng or np.random.default_rng()

    def transmit(self, encoded):
        if self.rng.random() < self.packet_loss:
            return None

        noise = self.rng.normal(
            loc=0.0,
            scale=self.noise_std,
            size=encoded.shape
        )

        return encoded + noise