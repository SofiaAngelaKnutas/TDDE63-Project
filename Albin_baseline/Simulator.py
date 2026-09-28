import numpy as np
from Albin_baseline.Classes.MarkovChain import MarkovChain
from Albin_baseline.Classes.Channel import Channel
from Albin_baseline.Classes.Decoder import Decoder
from Albin_baseline.Classes.Sender import Sender
import time

start = time.perf_counter()

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

T = 1000

source_rng = np.random.default_rng(42)
sender_rng = np.random.default_rng(43)
bob_rng = np.random.default_rng(44)
eve_rng = np.random.default_rng(45)

chain = MarkovChain(
    states=states,
    transition_matrix=P,
    rng=source_rng
)

sender = Sender(
    sending_probability=0.93, #0.93-0.94 found optimal for this simple test
    rng=sender_rng
)

# Bob has better channel
bob_channel = Channel(
    packet_loss=0.05,
    rng=bob_rng
)

# Eve has worse channel
eve_channel = Channel(
    packet_loss=0.25,
    rng=eve_rng
)

bob_decoder = Decoder(
    states=states
)

eve_decoder = Decoder(
    states=states
)

state_history = chain.sample(
    T=T,
    start_state=0
)

bob_prediction_Y = []
bob_prediction_S = []

eve_prediction_Y = []
eve_prediction_S = []


for t in range(T):
    true_index = state_history[t]   #e.g. "3" which corresponds to (1, 1)
    true_state = states[true_index] #e.g. true_index = 3 gives us (1, 1)

    sent_packet = sender.transmit(true_state)

    # Bob and Eve receive through different channels
    bob_received = bob_channel.observe(sent_packet)
    eve_received = eve_channel.observe(sent_packet)

    bob_estimate = bob_decoder.observe(bob_received)
    Y,S = bob_estimate
    bob_prediction_Y.append(Y)
    bob_prediction_S.append(S)

    eve_estimate = eve_decoder.observe(eve_received)
    Y,S = eve_estimate
    eve_prediction_Y.append(Y)
    eve_prediction_S.append(S)
    progress = (t + 1) / T * 100
    print(f"\rProgress: {progress:.1f}%", end="", flush=True)

print()
print("Simulation time: ", time.perf_counter() - start)

# Results:

bob_Y = np.array(bob_prediction_Y)
bob_S = np.array(bob_prediction_S)
eve_S = np.array(eve_prediction_S)

true_states = states[state_history]
true_Y = true_states[:, 0]
true_S = true_states[:, 1]

# Reconstruction correctness
bob_correct = (bob_Y == true_Y) & (bob_S == true_S)
eve_wrong = eve_S != true_S

CRA = bob_correct & eve_wrong

average_CRA = np.mean(CRA)

print("Bob joint (Y + S) accuracy:", np.mean(bob_correct))
print("Eve S accuracy:", np.mean(eve_S == true_S))
print("Average CRA:", average_CRA)