import numpy as np

class Decoder:
    """
    HMM/Bayesian filter.

    belief[i] = P(X_t = i | observations so far)
    """

    def __init__(
        self,
        transition_matrix,
        states,
        encoder,
        observation_noise_std,
        initial_belief=None
    ):
        self.P = np.asarray(transition_matrix, dtype=float)
        self.states = np.asarray(states)
        self.encoder = encoder
        self.noise_std = observation_noise_std

        self.codebook = encoder.codebook(states)

        n_states = len(states)

        if initial_belief is None:
            self.belief = np.ones(n_states) / n_states
        else:
            self.belief = np.asarray(initial_belief, dtype=float)
            self.belief /= self.belief.sum()

    def set_transition_matrix(self, P):
        """
        Useful for Eve because P' changes over time.
        """
        self.P = np.asarray(P, dtype=float)

    def predict(self):
        """
        Markov prediction:

        P(X_t | observations up to t-1)
        """
        self.belief = self.belief @ self.P

    def observation_likelihood(self, observation):
        """
        p(R_t | X_t = i)

        Constant Gaussian normalization factor omitted because
        same for every state.
        """
        difference = self.codebook - observation

        squared_distance = np.sum(difference ** 2, axis=1)

        log_likelihood = (
            -0.5
            * squared_distance
            / (self.noise_std ** 2)
        )

        # Numerical stability
        log_likelihood -= np.max(log_likelihood)

        return np.exp(log_likelihood)

    def update(self, observation):
        """
        Bayesian observation update.

        If packet is missing, keep prediction unchanged.
        """
        if observation is None:
            return

        likelihood = self.observation_likelihood(observation)

        posterior = self.belief * likelihood

        total = posterior.sum()

        if total == 0:
            self.belief = np.ones(len(self.states)) / len(self.states)
        else:
            self.belief = posterior / total

    def observe_initial(self, observation):
        """
        Observation at t=0: no Markov transition yet.
        """
        self.update(observation)

        return self.estimate()

    def step(self, observation):
        self.predict()
        self.update(observation)

        return self.estimate()

    def estimate(self):
        return np.argmax(self.belief)