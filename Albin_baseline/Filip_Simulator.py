import numpy as np
from Classes.MarkovChain import MarkovChain
from Classes.Channel import Channel
from Classes.Decoder import HMMDecoder
from Classes.Sender import Sender_Filip
from Classes.EveLearner import RandomForestEve
from Classes.TrainEve import RandomForest_Trainer
import time
from pathlib import Path

start = time.perf_counter()

#############################################
########### Tweak variables here: ###########
#############################################
PRETRAINED_MODEL = None #Set to None if we want to train new, otherwise the filename if we want to use a previous trained
#PRETRAINED_MODEL = "eve_random_forest.joblib"

# Ignore these if using pretrained model:
MODEL_NAME = "RF_Filip_Window4_Bob005_Eve025" + ".joblib"
TRAINING_RUNS = 5
T_PER_RUN = 20_000
WINDOW_SIZE = 4
OBSERVATION_SIZE = 2       # baseline cleartext packet = [Y, S], might need to change depending on defense method.

#
SENDING_PROBABILITY = 0.9  #0.93-0.94 found optimal for this simple test
BOB_PACKET_LOSS = 0.05
EVE_PACKET_LOSS = 0.25
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

source_rng = np.random.default_rng(42)
sender_rng = np.random.default_rng(43)
bob_rng = np.random.default_rng(44)
eve_rng = np.random.default_rng(45)

chain = MarkovChain(states=states, transition_matrix=P, rng=source_rng)

sender = Sender_Filip(
    states=states,
    transition_matrix=P,
    sending_probability=SENDING_PROBABILITY,
    rng=sender_rng
)

bob_channel = Channel(packet_loss=BOB_PACKET_LOSS, rng=bob_rng)
eve_channel = Channel(packet_loss=EVE_PACKET_LOSS, rng=eve_rng)

bob_decoder = HMMDecoder(states=states, transition_matrix=P)

if PRETRAINED_MODEL is None:

    training_sender = Sender_Filip(
        states=states,
        transition_matrix=P,
        sending_probability=SENDING_PROBABILITY,
        rng=np.random.default_rng(1337)
    )
    
    eve_trainer = RandomForest_Trainer(
        model_name=MODEL_NAME,
        states=states,
        P=P,
        training_runs=TRAINING_RUNS,
        T_per_run=T_PER_RUN,
        window_size=WINDOW_SIZE,
        observation_size=OBSERVATION_SIZE,
        eve_packet_loss=EVE_PACKET_LOSS,
        sender=training_sender,
    )

    eve_trainer.train()
    eve_decoder = RandomForestEve(
        model_path=Path("TempModels") / MODEL_NAME
    )
    
else:
    eve_decoder = RandomForestEve(
        model_path=Path("TempModels") / PRETRAINED_MODEL
    )

state_history = chain.sample(T=T,start_state=0)

bob_prediction_S = []
bob_prediction_Y = []

eve_feature_rows = []
eve_had_current_packet = []

for t in range(T):
    true_index = state_history[t]
    true_state = states[true_index]
    true_y, true_s = true_state

    sent_packet = sender.transmit(true_state)

    bob_received = bob_channel.observe(sent_packet)
    eve_received = eve_channel.observe(sent_packet)

    # Bob
    bob_y, bob_s = bob_decoder.observe(bob_received)

    bob_prediction_Y.append(bob_y)
    bob_prediction_S.append(bob_s)

    # Eve
    features = eve_decoder.observe(eve_received)
    eve_feature_rows.append(features)

    eve_had_current_packet.append(eve_received is not None)

    progress = (t + 1) / T * 100
    print(f"\rProgress: {progress:.1f}%", end="", flush=True)

print()
print("Simulation time: ", time.perf_counter() - start)

#Train all in one batch
eve_S = eve_decoder.predict(
    np.asarray(eve_feature_rows)
)

# Results:

bob_Y = np.array(bob_prediction_Y)
bob_S = np.array(bob_prediction_S)
eve_had_current_packet = np.asarray(eve_had_current_packet)

true_states = states[state_history]
true_Y = true_states[:, 0]
true_S = true_states[:, 1]

bob_correct = (bob_Y == true_Y) & (bob_S == true_S)
eve_wrong = eve_S != true_S

CRA = bob_correct & eve_wrong

print("Bob Y accuracy:", np.mean(bob_Y == true_Y))
print("Bob S accuracy:", np.mean(bob_S == true_S))
print("Bob joint (Y + S) accuracy:", np.mean(bob_correct))
print("Eve S accuracy:", np.mean(eve_S == true_S))
print("Average CRA:", np.mean(CRA))

# ~ is invert function. Basically, look at where Eve had no packet; 
# What was her accuracy then (so when the RF-model actually did some work)
hidden_mask = ~eve_had_current_packet
if np.any(hidden_mask):
    print(
        "Eve S accuracy when current packet unavailable:",
        np.mean(eve_S[hidden_mask] == true_S[hidden_mask]),
    )