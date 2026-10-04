import numpy as np


def randomized_response(true_index, theta, n_states, rng):
    """
    Randomized response defense, WITHOUT key.

    With probability theta, Alice sends the real state.
    Otherwise, she sends a random state (all states equally likely).

    true_index : index of the true state, e.g. trajectory[t]
    theta      : probability of sending the real state (0 to 1)
    n_states   : number of possible states (4 in our project)
    rng        : random number generator, e.g. np.random.default_rng(45)

    Returns the index of the state Alice actually sends.
    """
    
    send_real_state = rng.random() < theta

    if send_real_state:
        sent_index = true_index
    else:
        random_index = int(rng.integers(n_states))
        sent_index = random_index

    return sent_index