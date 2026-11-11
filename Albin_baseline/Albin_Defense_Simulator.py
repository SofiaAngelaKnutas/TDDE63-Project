import numpy as np
from Classes.MarkovChain import MarkovChain
from Classes.Channel import Channel
from Classes.Decoder import HMMDecoderAlbin
from Classes.Sender import Albin_Sender
from Classes.EveLearner import RandomForestEve
from Classes.TrainEve import RandomForestTrainer
from Classes.SimulationOptions import simulation_options
import time

start = time.perf_counter()

#############################################
########### Tweak variables here: ###########
#############################################
PRETRAINED_MODEL = None #Set to None if we want to train new, otherwise the filename if we want to use a previous trained
#PRETRAINED_MODEL = "eve_random_forest.joblib"

# Training defaults (ignored when loading a model):
DEFENSE = "albin"
MODEL_NAME = f"eve_{DEFENSE}.joblib"
TRAINING_RUNS = 5
T_PER_RUN = 2000

#
SENDING_PROBABILITY = 1.0
BOB_RECEPTION_PROBABILITY = 0.99
EVE_RECEPTION_PROBABILITY = 0.50
T = 100_000

BOB_WINDOW_SIZE = 1

states = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

#A much faster markov chain
P = np.array([
    [0.60, 0.10, 0.10, 0.20],
    [0.25, 0.35, 0.15, 0.25],
    [0.25, 0.15, 0.35, 0.25],
    [0.20, 0.10, 0.10, 0.60],
])
"""

P = np.array([
    [0.95, 0.01, 0.02, 0.02],
    [0.02, 0.96, 0.01, 0.01],
    [0.03, 0.02, 0.94, 0.01],
    [0.01, 0.02, 0.02, 0.95],
])
"""

options = simulation_options(DEFENSE, T, TRAINING_RUNS, T_PER_RUN)
T = options.steps
TRAINING_RUNS = options.training_runs
T_PER_RUN = options.training_steps
model_path = options.model_dir / MODEL_NAME
if options.model is not None:
    PRETRAINED_MODEL = options.model

source_rng = np.random.default_rng(42)
sender_rng = np.random.default_rng(43)
bob_rng = np.random.default_rng(44)
eve_rng = np.random.default_rng(45)

chain = MarkovChain(states=states, transition_matrix=P, rng=source_rng)

sender = Albin_Sender(
    states=states,
    Bob_window_size=BOB_WINDOW_SIZE,
    sending_probability=SENDING_PROBABILITY,
    rng=sender_rng
)

bob_channel = Channel(rec_probability=BOB_RECEPTION_PROBABILITY, rng=bob_rng)
eve_channel = Channel(rec_probability=EVE_RECEPTION_PROBABILITY, rng=eve_rng)

bob_decoder = HMMDecoderAlbin(states=states, transition_matrix=P, Bob_window_size=BOB_WINDOW_SIZE)

if PRETRAINED_MODEL is None:

    training_sender = Albin_Sender(
        states=states,
        Bob_window_size=BOB_WINDOW_SIZE,
        sending_probability=SENDING_PROBABILITY,
        rng=np.random.default_rng(1337)
    )
    
    eve_trainer = RandomForestTrainer(
        model_path=model_path,
        defense=DEFENSE,
        states=states,
        P=P,
        training_runs=TRAINING_RUNS,
        T_per_run=T_PER_RUN,
        eve_reception_probability=EVE_RECEPTION_PROBABILITY,
        bob_reception_probability=BOB_RECEPTION_PROBABILITY,
        sender=training_sender,
        feedback=lambda alice, index, ack: alice.update_Bob_window(index if ack else 0),
    )

    eve_trainer.train()
    eve_decoder = RandomForestEve(
        model_path=model_path, defense=DEFENSE
    )
    
else:
    eve_decoder = RandomForestEve(
        model_path=options.model_dir / PRETRAINED_MODEL, defense=DEFENSE
    )

if options.train_only:
    raise SystemExit(0)

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
    bob_state, ACK = bob_decoder.observe(bob_received)
    bob_y, bob_s = bob_state

    bob_prediction_Y.append(bob_y)
    bob_prediction_S.append(bob_s)

    # Eve
    features = eve_decoder.observe(eve_received)

    # Freeze this slot's features before exposing ACK_t or updating history.
    eve_feature_rows.append(features)
    eve_decoder.update(eve_received, ACK)
    # Feedback affects only future Alice transmissions.
    sender.update_Bob_window(true_index if ACK else 0)

    eve_had_current_packet.append(eve_received is not None)

    progress = (t + 1) / T * 100
    if not options.quiet:
        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
    #if progress > 99:
    #    print(f"Current features: ", features)


print()

# Eve is passive: predict all frozen feature rows in one batch.
eve_S = eve_decoder.predict(np.asarray(eve_feature_rows))
print(f"Simulation time: {time.perf_counter() - start:.1f} s")

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

print(f"Bob Y accuracy: {np.mean(bob_Y == true_Y):.1%}")
print(f"Bob S accuracy: {np.mean(bob_S == true_S):.1%}")
print(f"Bob joint (Y + S) accuracy: {np.mean(bob_correct):.1%}")
print(f"Eve S accuracy: {np.mean(eve_S == true_S):.1%}")
print(f"Average CRA: {np.mean(CRA):.1%}")

# ~ is invert function. Basically, look at where Eve had no packet; 
# With the current defense, this is NOT the same as when ML was doing work,
# Here the ML does work a lot of the time, but this is when nothing was received.
hidden_mask = ~eve_had_current_packet
if np.any(hidden_mask):
    print(f"Eve S accuracy when current packet unavailable: {np.mean(eve_S[hidden_mask] == true_S[hidden_mask]):.1%}")

received_mask = eve_had_current_packet

#Opposite of above, I think it could be interesting for this defense.
if np.any(received_mask):
    print(f"Eve S accuracy when encoded packet received: {np.mean(eve_S[received_mask] == true_S[received_mask]):.1%}")

values, counts = np.unique(eve_S, return_counts=True)

for value, count in zip(values, counts):
    print(f"Value {value}: {count:,} ({count / eve_S.size:.1%})")
