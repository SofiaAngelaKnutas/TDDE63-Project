"""
Receiver jamming defense, run against the group's ML Eve.

Same setup as Simulator_Simple.py, except the channels: Bob sends jamming noise
while he receives. Every packet's bits are turned into numbers (0 -> -1, 1 -> +1),
noise is added, and the receiver turns them back into bits. Eve gets the full
noise, Bob only a small leak, so Eve receives many wrong states and Bob few.

Bob uses Albin's HMMDecoder unchanged: he trusts every packet he receives. If a
packet is impossible (only happens at the very first step, when Bob knows the
chain starts in state 0), he ignores it, as if it never arrived.
Eve is trained the same way as in TrainEve.py (same window, same random forest),
but on traffic that already has the jamming in it, so she knows the defense.

Run from inside the Albin_baseline folder:
    python Simulator_Jamming.py
"""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Classes.MarkovChain import MarkovChain
from Classes.Decoder import HMMDecoder
from Classes.Sender import Sender
from Classes.EveLearner import ObservationWindow
from Classes.JammingChannel import JammingChannel

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

#############################################
########### Tweak variables here: ###########
#############################################
JAM_NOISE = [0.0, 0.5, 1.0, 2.0, 4.0]   # strength of Bob's jamming noise (0 = no jamming)
SELF_INTERFERENCE = 0.1                 # share of the noise power that leaks into Bob

# Same as Simulator_Simple.py:
BOB_PACKET_LOSS = 0.05
EVE_PACKET_LOSS = 0.25
TRAINING_RUNS = 5
T_PER_RUN = 20_000
WINDOW_SIZE = 4
OBSERVATION_SIZE = 2
T = 100_000

states = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

P = np.array([
    [0.60, 0.10, 0.10, 0.20],
    [0.25, 0.35, 0.15, 0.25],
    [0.25, 0.15, 0.35, 0.25],
    [0.20, 0.10, 0.10, 0.60],
])


def train_eve(jam_noise):
    """Train Eve like TrainEve.py, but on Eve's jammed channel."""
    master_rng = np.random.default_rng(1000)
    X_train, y_train = [], []
    for _ in range(TRAINING_RUNS):
        chain = MarkovChain(states, P, rng=np.random.default_rng(master_rng.integers(2**32)))
        eve_channel = JammingChannel(EVE_PACKET_LOSS, jam_noise, leak=1.0,
                                     rng=np.random.default_rng(master_rng.integers(2**32)))
        sender = Sender(sending_probability=1.0, rng=np.random.default_rng(1337))
        window = ObservationWindow(window_size=WINDOW_SIZE, observation_size=OBSERVATION_SIZE)
        state_history = chain.sample(T=T_PER_RUN, start_state=0)
        for t in range(T_PER_RUN):
            true_state = states[state_history[t]]
            eve_received = eve_channel.observe(sender.transmit(true_state))
            X_train.append(window.features(eve_received))
            y_train.append(int(true_state[1]))       # label: the private attribute S
            window.update(eve_received)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int))
    return model


def bob_observe(decoder, packet):
    """Albin's HMMDecoder, but an impossible packet is treated as a missing one."""
    first_slot = decoder.first_slot
    try:
        return decoder.observe(packet)
    except ValueError:                   # noise turned it into a state Bob knows is impossible
        decoder.first_slot = first_slot
        return decoder.observe(None)


def run(jam_noise):
    bob_channel = JammingChannel(BOB_PACKET_LOSS, jam_noise, leak=SELF_INTERFERENCE,
                                 rng=np.random.default_rng(44))
    eve_channel = JammingChannel(EVE_PACKET_LOSS, jam_noise, leak=1.0,
                                 rng=np.random.default_rng(45))

    eve_model = train_eve(jam_noise)
    eve_window = ObservationWindow(window_size=WINDOW_SIZE, observation_size=OBSERVATION_SIZE)
    bob_decoder = HMMDecoder(states=states, transition_matrix=P)

    chain = MarkovChain(states=states, transition_matrix=P, rng=np.random.default_rng(42))
    sender = Sender(sending_probability=1.0, rng=np.random.default_rng(43))
    state_history = chain.sample(T=T, start_state=0)

    bob_predictions, eve_feature_rows = [], []
    for t in range(T):
        sent_packet = sender.transmit(states[state_history[t]])
        bob_predictions.append(bob_observe(bob_decoder, bob_channel.observe(sent_packet)))
        eve_received = eve_channel.observe(sent_packet)
        eve_feature_rows.append(eve_window.features(eve_received))
        eve_window.update(eve_received)

    eve_S = eve_model.predict(np.asarray(eve_feature_rows, dtype=float))
    true_states = states[state_history]
    bob_correct = np.all(np.array(bob_predictions) == true_states, axis=1)
    eve_wrong = eve_S != true_states[:, 1]
    return {
        "bob_bit_err": bob_channel.bit_error_prob,
        "eve_bit_err": eve_channel.bit_error_prob,
        "bob_acc": np.mean(bob_correct),
        "eve_S_acc": 1 - np.mean(eve_wrong),
        "CRA": np.mean(bob_correct & eve_wrong),
    }


if __name__ == "__main__":
    print(f"Packet loss without jamming: Bob {BOB_PACKET_LOSS}, Eve {EVE_PACKET_LOSS}. "
          f"Self-interference at Bob: {SELF_INTERFERENCE}\n")
    print(f"{'noise':>5} | {'Bob bit err':>11} {'Eve bit err':>11} | {'Bob acc':>7} {'Eve S acc':>9} {'CRA':>6}")
    print("-" * 62)
    for noise in JAM_NOISE:
        r = run(noise)
        print(f"{noise:>5} | {r['bob_bit_err']:>11.3f} {r['eve_bit_err']:>11.3f} | "
              f"{r['bob_acc']:>7.3f} {r['eve_S_acc']:>9.3f} {r['CRA']:>6.3f}")
