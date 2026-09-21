import numpy as np

class TransitionLearner:
    """
    Eve's estimate P'.

    Uses Dirichlet/Laplace smoothing.

    alpha = 1 means Eve starts with:
        [0.25, 0.25, 0.25, 0.25]
    for every row.
    """

    def __init__(self, n_states, alpha=1.0):
        self.n_states = n_states

        self.counts = np.full(
            (n_states, n_states),
            alpha,
            dtype=float
        )

    def update(self, previous_state, current_state):
        self.counts[previous_state, current_state] += 1

    @property
    def P(self):
        return self.counts / self.counts.sum(
            axis=1,
            keepdims=True
        )