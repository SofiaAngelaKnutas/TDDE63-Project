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
        
        if self.rng.random() > self.sending_probability:
            return None
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
    def __init__(self, theta, states, sending_probability=1.0, rng=None):
        self.sending_probability = sending_probability
        self.theta = theta
        self.states = states
        self.rng = rng or np.random.default_rng()
        
    def transmit(self, packet):

        send_real_state = self.rng.random() < self.theta
        if self.rng.random() > self.sending_probability:
            return None
        if send_real_state:
            return packet
        else:
            random_index = self.rng.integers(len(self.states)) #A random state
            return self.states[random_index].copy()

class Sofia_Keyed_Sender:
    """
    Randomized response defense, WITH key.

    Alice and Bob share a secret key (a seed). Both use it to create
    the same random generator, so they make the same real/fake
    decision in every slot. Eve does not have the key.

    theta      : probability of sending the real state (0 to 1)
    states     : all possible states
    key_seed   : the shared secret, known only by Alice and Bob
    rng        : random generator for picking the fake state (not secret)
    """
    def __init__(self, theta, states, key_seed, sending_probability=1.0, rng=None):
        self.sending_probability = sending_probability
        self.theta = theta
        self.states = states
        self.key_rng = np.random.default_rng(key_seed)
        self.rng = rng or np.random.default_rng()

    def transmit(self, packet):
        send_real_state = self.key_rng.random() < self.theta
        if self.rng.random() > self.sending_probability:
            return None
        if send_real_state:
            return packet
        else:
            random_index = self.rng.integers(len(self.states))
            return self.states[random_index].copy()


class Albin_Sender:
    def __init__(self, states, Bob_window_size, sending_probability=1.0, rng=None):
        self.sending_probability = sending_probability
        self.states = states
        self.Bob_window_size = Bob_window_size
        self.rng = rng or np.random.default_rng()

        self.state_to_idx = { #Just to translate [1, 1] == 3 for example
            tuple(state): i
            for i, state in enumerate(self.states)
        }

        self.Bob_window = deque(
            [0] * (self.Bob_window_size),
            maxlen=Bob_window_size,
        )

    def transmit(self, packet):
        history_sum = sum(self.Bob_window)
        current_idx = self.state_to_idx[tuple(packet)]
        packet_idx = (current_idx - history_sum) % len(self.states)
        packet_to_transmit = self.states[packet_idx]
        if self.rng.random() > self.sending_probability:
            return None        
        return packet_to_transmit

    def update_Bob_window(self, Bob_received_packet_idx):
        self.Bob_window.appendleft(Bob_received_packet_idx)

    def reset(self):
        self.Bob_window.clear()
        self.Bob_window.extend(
            [0] * self.Bob_window_size
        )

class Sender_Emelie:
    """Encode against the last ACKed source state, without a window.

    Feedback is reliable and arrives after this slot, before the next transmit.
    An absent ACK leaves the reference unchanged, including across loss bursts.
    """
    def __init__(self, states, sending_probability=1.0, rng=None):
        self.sending_probability = sending_probability
        self.states = np.asarray(states)
        self.rng = rng or np.random.default_rng()

        self.state_to_idx = { #Just to translate [1, 1] == 3 for example
            tuple(state): i
            for i, state in enumerate(self.states)
        }

        self.reference = 0 #initial R for Alice and Bob
        self.pending_state_idx = None

    def transmit(self, packet):
        current_idx = self.state_to_idx[tuple(packet)]
        packet_idx = (current_idx + self.reference) % len(self.states)
        # Alice already knows her own source state; ACK carries only success.
        self.pending_state_idx = current_idx
        if self.rng.random() > self.sending_probability:
            return None
        return self.states[packet_idx].copy() #copy incase other code modifies it (might not be needed)

    def ack(self, state_idx=None):
        """ACK the current transmission, never a prediction made during loss.

        The optional index preserves the existing ack(state_idx) interface.
        It must match the state Alice actually transmitted in this slot.
        """
        if self.pending_state_idx is None:
            raise ValueError("No transmission awaiting ACK")
        if state_idx is not None and state_idx != self.pending_state_idx:
            raise ValueError("ACK does not match the current transmission")
        self.reference = self.pending_state_idx
        self.pending_state_idx = None

    def reset(self):
        self.reference = 0
        self.pending_state_idx = None
