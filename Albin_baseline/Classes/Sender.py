import numpy as np

class Sender:
    def __init__(self, sending_probability=1.0, rng=None):
        self.sending_probability = sending_probability
        self.rng = rng or np.random.default_rng()

    def transmit(self, packet):
        if self.rng.random() > self.sending_probability:
            return None

        return packet

class Sender_Filip:
    def __init__(self, states, transition_matrix, sending_probability=1.0, rng=None):
            self.sending_probability = sending_probability
            self.rng = rng or np.random.default_rng()

            self.states = np.asarray(states)
            self.P = np.asarray(transition_matrix, dtype=float)

            self.state_to_idx = { #Just to translate [1, 1] == 3 for example
                tuple(state): i
                for i, state in enumerate(self.states)
            }

            self.previous_state = self.states[0].copy()
    
    def transmit(self, packet):
        previous_idx = self.state_to_idx[tuple(self.previous_state)]
        expected_state_array = self.P[previous_idx]
        expected_state_idx = np.argmax(expected_state_array)
        expected_state = self.states[expected_state_idx]

        self.previous_state = packet.copy()

        # If current state is exactly what was expected,
        # don't bother transmitting
        if np.array_equal(expected_state, packet):
            return None

        return packet