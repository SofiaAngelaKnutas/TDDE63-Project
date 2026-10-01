"""Bob's lossy channel and Markov-state prediction."""

import numpy as np


class Bob:
    """Authorized receiver that predicts the next state when no packet arrives."""

    def __init__(
        self,
        states,
        transition_matrix,
        packet_loss=0.05,
        rng=None,
        initial_estimate=None,
    ):
        self.states = np.asarray(states)
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        self._validate_model()

        if not 0.0 <= packet_loss <= 1.0:
            raise ValueError("packet_loss must be between 0 and 1")
        self.packet_loss = packet_loss
        self.rng = rng or np.random.default_rng()
        self.estimated_state_index = (
            0 if initial_estimate is None else self._state_index(initial_estimate)
        )
        self.previous_estimated_state = self.states[self.estimated_state_index].copy()
        self.packets_lost = 0

    def receive_packet(self, packet):
        """Apply channel loss and pass the received value to ``predict``."""
        if packet is not None and self.rng.random() < self.packet_loss:
            self.packets_lost += 1
            packet = None
        return self.predict(packet)

    def predict(self, packet):
        """Use a packet directly, or predict the MAP successor on no packet."""
        if packet is not None:
            self.estimated_state_index = self._state_index(packet)
        else:
            self.estimated_state_index = int(
                np.argmax(self.transition_matrix[self.estimated_state_index])
            )

        self.previous_estimated_state = self.states[self.estimated_state_index].copy()
        return self.previous_estimated_state.copy()

    def _state_index(self, state):
        state = np.asarray(state)
        if state.shape != self.states.shape[1:]:
            raise ValueError(f"state must have shape {self.states.shape[1:]}")
        matches = np.flatnonzero(np.all(self.states == state, axis=1))
        if len(matches) != 1:
            raise ValueError(f"state {state.tolist()} is not in Bob's state list")
        return int(matches[0])

    def _validate_model(self):
        if self.states.ndim != 2 or len(self.states) == 0:
            raise ValueError("states must be a non-empty 2D array")
        if self.transition_matrix.shape != (len(self.states), len(self.states)):
            raise ValueError("transition matrix shape must match the number of states")
        if np.any(self.transition_matrix < 0) or not np.all(np.isfinite(self.transition_matrix)):
            raise ValueError("transition probabilities must be finite and non-negative")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise ValueError("each transition-matrix row must sum to 1")
