import numpy as np

class Sender:
    def __init__(self, sending_probability=1.0, rng=None):
        self.sending_probability = sending_probability
        self.rng = rng or np.random.default_rng()

    def transmit(self, packet):
        if self.rng.random() > self.sending_probability:
            return None

        return packet