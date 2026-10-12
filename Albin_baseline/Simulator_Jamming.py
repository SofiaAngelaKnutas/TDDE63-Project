"""
Receiver jamming defense (idea from Zheng et al., IEEE TSP 2013).

The defense itself is in Classes/JammingChannel.py: it calculates the chance
that a packet gets through to Bob and to Eve when Bob jams. Jamming lowers
Eve's chance a lot, and Bob's only a little (the part of his own jamming he
cannot cancel). This file tests it for several jamming strengths.

Everything else is Albin's baseline: Alice sends every state, Bob uses the HMM
decoder, Eve is the random forest. Eve is trained with the same jamming, so she
knows about the defense.

Run from inside the Albin_baseline folder:
    python Simulator_Jamming.py
"""
from pathlib import Path

import numpy as np

from Classes.MarkovChain import MarkovChain
from Classes.Channel import Channel
from Classes.Decoder import HMMDecoder
from Classes.Sender import Sender
from Classes.EveLearner import RandomForestEve
from Classes.TrainEve import RandomForest_Trainer
from Classes.JammingChannel import jamming_probabilities

#############################################
########### Tweak variables here: ###########
#############################################
JAM_POWERS = [0, 1, 3, 10, 30]  # how strong Bob's jamming is (0 = no jamming)
SELF_INTERFERENCE = 0.1         # share of Bob's jamming he can NOT cancel (0 = perfect)

# Same as Simulator_Simple.py:
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


def run(jam_power):
    # 1. Calculate the chances and make the two channels
    p_ab, p_ae = jamming_probabilities(jam_power, SELF_INTERFERENCE)   # THE DEFENSE
    bob_channel = Channel(packet_loss=1 - p_ab, rng=np.random.default_rng(44))
    eve_channel = Channel(packet_loss=1 - p_ae, rng=np.random.default_rng(45))

    # 2. Train Eve on traffic with the same jamming (Albin's trainer)
    model_name = f"RF_Jamming_{jam_power}.joblib"
    RandomForest_Trainer(
        model_name=model_name, states=states, P=P,
        training_runs=TRAINING_RUNS, T_per_run=T_PER_RUN,
        window_size=WINDOW_SIZE, observation_size=OBSERVATION_SIZE,
        eve_packet_loss=1 - p_ae,
        sender=Sender(sending_probability=1.0, rng=np.random.default_rng(1337)),
    ).train()
    eve = RandomForestEve(model_path=Path("TempModels") / model_name)

    # 3. Alice, Bob and the true states
    sender = Sender(sending_probability=1.0, rng=np.random.default_rng(43))
    bob = HMMDecoder(states=states, transition_matrix=P)
    chain = MarkovChain(states=states, transition_matrix=P, rng=np.random.default_rng(42))
    state_history = chain.sample(T=T, start_state=0)

    # 4. Send T packets
    bob_guesses, eve_features, bob_got, eve_got = [], [], [], []
    for t in range(T):
        packet = sender.transmit(states[state_history[t]])
        bob_received = bob_channel.observe(packet)
        eve_received = eve_channel.observe(packet)

        bob_guesses.append(bob.observe(bob_received))
        eve_features.append(eve.observe(eve_received))
        bob_got.append(bob_received is not None)
        eve_got.append(eve_received is not None)

    # 5. Grade Bob and Eve against the true states
    true_states = states[state_history]
    eve_S = eve.predict(np.asarray(eve_features))
    bob_correct = np.all(np.array(bob_guesses) == true_states, axis=1)
    eve_wrong = eve_S != true_states[:, 1]
    bob_got, eve_got = np.array(bob_got), np.array(eve_got)

    return {
        "P_AB": p_ab,
        "P_AE": p_ae,
        "B_and_not_E": np.mean(bob_got & ~eve_got),   # Bob got it AND Eve did not
        "bob_acc": np.mean(bob_correct),
        "eve_S_acc": 1 - np.mean(eve_wrong),
        "CRA": np.mean(bob_correct & eve_wrong),
    }


if __name__ == "__main__":
    results = {jam: run(jam) for jam in JAM_POWERS}
    print()
    print(f"{'jam':>4} | {'P_AB':>5} {'P_AE':>5} | {'Bob yes, Eve no':>15} | "
          f"{'Bob acc':>7} {'Eve S acc':>9} {'CRA':>6}")
    print("-" * 64)
    for jam, r in results.items():
        print(f"{jam:>4} | {r['P_AB']:>5.3f} {r['P_AE']:>5.3f} | {r['B_and_not_E']:>15.3f} | "
              f"{r['bob_acc']:>7.3f} {r['eve_S_acc']:>9.3f} {r['CRA']:>6.3f}")
