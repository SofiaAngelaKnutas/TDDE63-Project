import numpy as np

class Encoder:
    """
    Deterministic part of encoder.

    Input:
        state = [Y, S] with values 0 or 1

    Internally convert to:
        0 -> -1
        1 -> +1
    """

    def encode(self, state):
        Y, S = state

        Y = 2 * Y - 1
        S = 2 * S - 1

        return np.array([
            Y,
            0.8 * Y,
            S,
            0.8 * S,
            0.7 * Y + 0.3 * S,
            0.3 * Y + 0.7 * S,
            Y * S,
            0.5 * Y - 0.5 * S,
        ], dtype=float)

    def codebook(self, states):
        """
        Expected encoded vector for every possible state.
        """
        return np.array([
            self.encode(state)
            for state in states
        ])