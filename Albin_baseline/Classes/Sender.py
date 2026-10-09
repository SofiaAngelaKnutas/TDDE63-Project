import numpy as np
from collections import deque

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

class Sofia_Random_Sender:
    """
    Randomized response defense, WITHOUT key.

    With probability theta, Alice sends the real state.
    Otherwise, she sends a random state (all states equally likely).

    theta      : probability of sending the real state (0 to 1)
    states     : all current possible states
    rng        : random number generator, e.g. np.random.default_rng(45)

    Returns the index of the state Alice actually sends.
    """
    def __init__(self, theta, states, rng):
        self.theta = theta
        self.states = states
        self.rng = rng or np.random.default_rng()
        
    def transmit(self, packet):

        send_real_state = self.rng.random() < self.theta

        if send_real_state:
            return packet
        else:
            random_index = self.rng.integers(len(self.states)) #A random state
            return self.states[random_index].copy()


class Albin_Sender:
    def __init__(self, states, Bob_window_size, rng=None):
        self.states = states
        self.Bob_window_size = Bob_window_size
        self.rng = rng or np.random.default_rng()

        self.state_to_idx = { #Just to translate [1, 1] == 3 for example
            tuple(state): i
            for i, state in enumerate(self.states)
        }

        self.BobWindow = deque(
            0 * (self.Bob_window_size),
            maxlen=Bob_window_size,
        )


    def transmit(self, packet):
        sum = sum(
            value if i % 2 == 0 else -value #+ - + - ...
            for i, value in enumerate(self.Bob_window_size)
        )
        modulo = sum % 4
        current_idx = self.state_to_idx[tuple(packet)]
        packet_to_transmit = (current_idx - modulo) % 4

        return packet_to_transmit

    

        