"""
Illustrative Markov model for the Filip_protocol simulation.

State tuple order is (Y, S):
    Y: public weather attribute (0 = no rain, 1 = rain)
    S: private soldier health attribute (0 = sick, 1 = healthy)

The transition probabilities below are simple example values for development.
They are not estimated from data and should not be interpreted as realistic
weather or health probabilities.
"""

import numpy as np


# Rows and columns use this same state order:
#   0: (0, 0) no rain, sick      -> soldier in house 1
#   1: (0, 1) no rain, healthy   -> soldier in house 2
#   2: (1, 0) rain, sick         -> soldier in house 3
#   3: (1, 1) rain, healthy      -> soldier in house 4
STATES = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=int,
)


# TRANSITION_MATRIX[i, j] is P(next state = STATES[j] | current state = STATES[i]).
# Each row sums to 1. Some rows have a different most probable next state.
TRANSITION_MATRIX = np.array(
    [
        [0.20, 0.50, 0.20, 0.10],  # From (0, 0), most likely next: (0, 1)
        [0.15, 0.45, 0.10, 0.30],  # From (0, 1), most likely next: (0, 1)
        [0.10, 0.20, 0.20, 0.50],  # From (1, 0), most likely next: (1, 1)
        [0.10, 0.40, 0.20, 0.30],  # From (1, 1), most likely next: (0, 1)
    ],
    dtype=float,
)
