"""Bob variant that reconstructs MAP-predicted states on silence."""

import numpy as np

from Filip_protocol.General.Bob import Bob as BaseBob


class Bob(BaseBob):
    """Use received states directly and infer the MAP successor on silence."""

    def predict(self, packet):
        if packet is None:
            self.estimated_state_index = int(
                np.argmax(self.transition_matrix[self.estimated_state_index])
            )
        else:
            self.estimated_state_index = self._state_index(packet)

        self.previous_estimated_state = self.states[self.estimated_state_index].copy()
        return self.previous_estimated_state.copy()
