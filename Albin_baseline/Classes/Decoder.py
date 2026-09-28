import numpy as np

class Decoder:
    def __init__(
            self,
            states,
        ):
        self.states = np.asarray(states)
        self.previous_estimated_state = [0, 0]

    def observe(self, transmitted_packet):
        if transmitted_packet is None: #packet missing
            return self.previous_estimated_state
        else:
            self.previous_estimated_state = transmitted_packet
            return transmitted_packet