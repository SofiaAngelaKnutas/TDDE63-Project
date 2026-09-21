import numpy as np
from Classes.MarkovChain import MarkovChain
from Classes.Encoder import Encoder
from Classes.Channel import Channel
from Classes.Decoder import Decoder
from Classes.TransitionLearner import TransitionLearner

states = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

P = np.array([
    [0.60, 0.15, 0.10, 0.15],
    [0.20, 0.50, 0.20, 0.10],
    [0.10, 0.20, 0.50, 0.20],
    [0.15, 0.10, 0.15, 0.60],
])

T = 100

source_rng = np.random.default_rng(42)
bob_rng = np.random.default_rng(43)
eve_rng = np.random.default_rng(44)

chain = MarkovChain(
    states=states,
    transition_matrix=P,
    rng=source_rng
)

encoder = Encoder()

# Bob has better channel
bob_channel = Channel(
    noise_std=0.4,
    packet_loss=0.05,
    rng=bob_rng
)

# Eve has worse channel
eve_channel = Channel(
    noise_std=0.8,
    packet_loss=0.40,
    rng=eve_rng
)

trajectory = chain.sample(
    T=T,
    start_state=0
)

# Bob knows exact P
bob_decoder = Decoder(
    transition_matrix=P,
    states=states,
    encoder=encoder,
    observation_noise_std=bob_channel.noise_std
)

# Eve starts without knowing P
eve_learner = TransitionLearner(
    n_states=len(states),
    alpha=1.0
)

eve_decoder = Decoder(
    transition_matrix=eve_learner.P,
    states=states,
    encoder=encoder,
    observation_noise_std=eve_channel.noise_std
)

bob_predictions = []
eve_predictions = []

P_errors = []


for t in range(T):
    true_index = trajectory[t]
    true_state = states[true_index]

    # Alice encodes current state
    encoded = encoder.encode(true_state)

    # Bob and Eve receive through different channels
    bob_received = bob_channel.transmit(encoded)
    eve_received = eve_channel.transmit(encoded)

    # Eve uses only P' learned BEFORE current state is revealed
    eve_decoder.set_transition_matrix(eve_learner.P)

    if t == 0:
        bob_estimate = bob_decoder.observe_initial(
            bob_received
        )

        eve_estimate = eve_decoder.observe_initial(
            eve_received
        )

    else:
        bob_estimate = bob_decoder.step(
            bob_received
        )

        eve_estimate = eve_decoder.step(
            eve_received
        )

    bob_predictions.append(bob_estimate)
    eve_predictions.append(eve_estimate)

    # --------------------------------------------------
    # AFTER decoding, previous/current state becomes public.
    # Eve can now learn this transition for future packets.
    # --------------------------------------------------

    if t > 0:
        eve_learner.update(
            trajectory[t - 1],
            trajectory[t]
        )

    # Measure how close Eve's P' is to true P
    error = np.linalg.norm(
        P - eve_learner.P,
        ord="fro"
    )

    P_errors.append(error)

# Results:

bob_predictions = np.array(bob_predictions)
eve_predictions = np.array(eve_predictions)

bob_accuracy = np.mean(
    bob_predictions == trajectory
)

eve_accuracy = np.mean(
    eve_predictions == trajectory
)

print("Bob accuracy:", bob_accuracy)
print("Eve accuracy:", eve_accuracy)

print("\nTrue P:")
print(P)

print("\nEve's learned P':")
print(np.round(eve_learner.P, 3))

print("\nFinal ||P - P'||:")
print(P_errors[-1])