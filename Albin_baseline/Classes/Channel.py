import numpy as np

class Channel:

    def __init__(self, packet_loss=0.0, rng=None):
        self.packet_loss = packet_loss
        self.rng = rng or np.random.default_rng()

    def observe(self, packet):
        if self.rng.random() < self.packet_loss:
            return None

        return packet