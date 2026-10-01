import numpy as np

class Decoder:
    """
    Old, we dont use this anymore
    """
    def __init__(
            self,
            states,
        ):
        self.states = np.asarray(states)
        self.previous_estimated_state = [0, 0]

    def observe(self, received_packet):
        if received_packet is None: #packet missing
            return self.previous_estimated_state
        else:
            self.previous_estimated_state = received_packet
            return received_packet

class HMMDecoder:
    def __init__(self, states, transition_matrix):
        self.states = np.asarray(states)
        self.P = np.asarray(transition_matrix, dtype=float)

        # Initial belief: Bob knows simulation starts in state 0
        self.belief = np.zeros(len(self.states))
        self.belief[0] = 1.0

        self.first_slot = True

    def observe(self, received_packet):
        # STEP 1: Predict current state

        if self.first_slot:
            # At t=0, source is already in initial state.
            prior = self.belief.copy()
            self.first_slot = False
        else:
            # Predict next state using Markov transition matrix
            prior = self.belief @ self.P

        # STEP 2: Calculate observation likelihood

        if received_packet is None:
            # Missing packet provides no new state information
            likelihood = np.ones(len(self.states))
        else:
            # Perfect reception: only matching state is possible
            likelihood = np.all(
                self.states == received_packet, axis=1
            ).astype(float)

        # STEP 3: Bayesian update (Basically: (what markov said) * (what the received packet said))
        posterior = prior * likelihood

        total = posterior.sum()
        if total == 0:
            raise ValueError("Received observation has zero probability")

        self.belief = posterior / total

        # STEP 4: Return most probable state (MAP)
        estimated_index = np.argmax(self.belief)

        return self.states[estimated_index].copy()