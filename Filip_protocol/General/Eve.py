"""Eve's lossy channel and prediction without the private transition matrix."""

import numpy as np


class Eve:
    """Eavesdropper that uses received states and holds her last estimate."""

    def __init__(self, packet_loss=0.25, rng=None, initial_estimate=(0, 0)):
        if not 0.0 <= packet_loss <= 1.0:
            raise ValueError("packet_loss must be between 0 and 1")
        self.packet_loss = packet_loss
        self.rng = rng or np.random.default_rng()
        self.previous_estimated_state = np.asarray(initial_estimate, dtype=int).copy()
        self.packets_lost = 0

    def receive_packet(self, packet):
        """Apply Eve's channel loss, then pass the observation to ``predict``."""
        if packet is not None and self.rng.random() < self.packet_loss:
            self.packets_lost += 1
            packet = None
        return self.predict(packet)

    def predict(self, packet):
        """Copy a received state; without one, retain the last observed state.

        Eve has no transition matrix in this threat model, so she cannot make
        a Markov-based prediction when a packet is missing.
        """
        if packet is not None:
            self.previous_estimated_state = np.asarray(packet, dtype=int).copy()
        return self.previous_estimated_state.copy()
