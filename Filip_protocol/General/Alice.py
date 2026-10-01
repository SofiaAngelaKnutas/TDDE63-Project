"""Alice creates the Markov source sequence and sends its states."""

import numpy as np


class Alice:
    """Creates the source sequence and probabilistically sends each state."""

    def __init__(
        self,
        states,
        transition_matrix,
        sequence_length,
        sending_probability=0.93,
        start_state_index=0,
        source_rng=None,
        sender_rng=None,
    ):
        self.states = np.asarray(states)
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        self.sequence_length = sequence_length
        self.sending_probability = sending_probability
        self.source_rng = source_rng or np.random.default_rng()
        self.sender_rng = sender_rng or np.random.default_rng()

        #Felhantering start

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if self.states.ndim != 2 or len(self.states) == 0:
            raise ValueError("states must be a non-empty 2D array")
        if self.transition_matrix.shape != (len(self.states), len(self.states)):
            raise ValueError("transition matrix shape must match the number of states")
        if np.any(self.transition_matrix < 0) or not np.all(np.isfinite(self.transition_matrix)):
            raise ValueError("transition probabilities must be finite and non-negative")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise ValueError("each transition-matrix row must sum to 1")
        if not 0.0 <= self.sending_probability <= 1.0:
            raise ValueError("sending_probability must be between 0 and 1")
        if not 0 <= start_state_index < len(self.states):
            raise ValueError("start_state_index is outside the state list")

        #Felhantering slut

        self.sequence_indices = np.empty(self.sequence_length, dtype=int)
        self.sequence_indices[0] = start_state_index
        for step in range(1, self.sequence_length):
            previous_index = self.sequence_indices[step - 1]
            self.sequence_indices[step] = self.source_rng.choice(
                len(self.states), p=self.transition_matrix[previous_index]
            )
        self.sequence = self.states[self.sequence_indices]
        self.packets_sent = 0

    def send_packet(self, step):
        """Return this time step's full state, or None if Alice skips it."""
        if not 0 <= step < self.sequence_length:
            raise IndexError("step is outside Alice's generated sequence")
        if self.sender_rng.random() > self.sending_probability:
            return None

        self.packets_sent += 1
        return self.sequence[step].copy()
