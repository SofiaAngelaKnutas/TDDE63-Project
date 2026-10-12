import numpy as np
from collections import deque

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

class HMMDecoderSofia:
    """
    Because Sofia's random defence can send random states, Bob must change his
    likelihood thinking. If he sees packet [1,1] he cannot set likelihood of
    state [1,1] to 100%. He must take theta into consideration aswell.
    """
    def __init__(self, states, transition_matrix, theta):
        self.states = np.asarray(states)
        self.P = np.asarray(transition_matrix, dtype=float)

        # Initial belief: Bob knows simulation starts in state 0
        self.belief = np.zeros(len(self.states))
        self.belief[0] = 1.0

        self.first_slot = True

        self.theta = theta


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
            n_states = len(self.states)
            random_probability = (1 - self.theta) / n_states

            likelihood = np.full(
                n_states,
                random_probability
            )
            
            matches = np.all(
                self.states == received_packet,
                axis=1
            )

            likelihood[matches] += self.theta

        # STEP 3: Bayesian update (Basically: (what markov said) * (what the received packet said))
        posterior = prior * likelihood

        total = posterior.sum()
        if total == 0:
            raise ValueError("Received observation has zero probability")

        self.belief = posterior / total

        # STEP 4: Return most probable state (MAP)
        estimated_index = np.argmax(self.belief)

        return self.states[estimated_index].copy()

class HMMDecoderAlbin:
    def __init__(self, states, transition_matrix, Bob_window_size):
        self.states = np.asarray(states)
        self.P = np.asarray(transition_matrix, dtype=float)

        # Initial belief: Bob knows simulation starts in state 0
        self.belief = np.zeros(len(self.states))
        self.belief[0] = 1.0

        self.first_slot = True

        self.Bob_window_size = Bob_window_size
        self.Bob_window = deque(
            [0] * (self.Bob_window_size),
            maxlen=Bob_window_size,
        )

        self.state_to_idx = { #Just to translate [1, 1] == 3 for example
            tuple(state): i
            for i, state in enumerate(self.states)
        }

    def observe(self, received_packet):
        # STEP 1: Predict current state
        ACK = False

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
            self.Bob_window.appendleft(0)
            ACK = False
        else:
            # Perfect reception: Decode the state then update likelihood + window + ACK
            current_idx = self.state_to_idx[tuple(received_packet)]
            history_sum = sum(self.Bob_window)
            decoded_packet = (current_idx + history_sum) % len(self.states)

            likelihood = np.zeros(len(self.states))
            likelihood[decoded_packet] = 1.0
            self.Bob_window.appendleft(decoded_packet)
            ACK = True

        # STEP 3: Bayesian update (Basically: (what markov said) * (what the received packet said))
        posterior = prior * likelihood

        total = posterior.sum()
        if total == 0:
            raise ValueError("Received observation has zero probability")

        self.belief = posterior / total

        # STEP 4: Return most probable state (MAP)
        estimated_index = np.argmax(self.belief)

        return self.states[estimated_index].copy(), ACK