"""Alice variant that suppresses packets for MAP-predicted transitions."""

import numpy as np

from Filip_protocol.General.Alice import Alice as BaseAlice


class Alice(BaseAlice):
    """Send only when the next state differs from the MAP prediction."""

    def send_packet(self, step):
        if not 0 <= step < self.sequence_length:
            raise IndexError("step is outside Alice's generated sequence")
        if step == 0:
            # The first state is the shared initial condition.
            return None

        previous_state_index = self.sequence_indices[step - 1]
        predicted_state_index = int(
            np.argmax(self.transition_matrix[previous_state_index])
        )
        actual_state_index = self.sequence_indices[step]

        if actual_state_index == predicted_state_index:
            return None

        self.packets_sent += 1
        return self.sequence[step].copy()
