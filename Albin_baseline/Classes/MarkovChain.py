import numpy as np

class MarkovChain:
    def __init__(self, states, transition_matrix, rng=None):
        self.states = np.asarray(states)
        self.P = np.asarray(transition_matrix, dtype=float)
        self.rng = rng or np.random.default_rng()

        n = len(self.states)

        if self.P.shape != (n, n):
            raise ValueError("Transition matrix has wrong shape")

        if not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError("Each row of transition matrix must sum to 1")

    def sample(self, T, start_state=0):
        indices = np.empty(T, dtype=int)
        indices[0] = start_state

        for t in range(1, T):
            indices[t] = self.rng.choice(
                len(self.states),
                p=self.P[indices[t - 1]]
            )

        return indices