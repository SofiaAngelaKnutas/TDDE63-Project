from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Classes.MarkovChain import MarkovChain
from Classes.Channel import Channel
from Classes.EveLearner import ObservationWindow, FEATURE_COUNT, SCHEMA_VERSION


class RandomForestTrainer:
    def __init__(self, model_path, defense, states, P, training_runs, T_per_run,
                 eve_reception_probability, bob_reception_probability, sender, feedback=None, seed=1000):
        self.model_path = Path(model_path)
        self.defense = defense
        self.states = np.asarray(states)
        self.P = np.asarray(P)
        if training_runs < 1 or T_per_run < 1:
            raise ValueError("Training runs and steps must be positive")
        self.training_runs = training_runs
        self.T_per_run = T_per_run
        self.eve_reception_probability = eve_reception_probability
        self.bob_reception_probability = bob_reception_probability
        self.sender = deepcopy(sender) #Creates fresh copy without memory from last run
        self.feedback = feedback
        self.seed = seed

    def generate_data(self):
        X, y = [], []
        # Separate streams for source, sender, Bob, Eve
        run_seeds = np.random.SeedSequence(self.seed).spawn(self.training_runs)
        for run_seed in run_seeds:
            source_seed, sender_seed, bob_seed, eve_seed, key_seed = run_seed.spawn(5)
            chain = MarkovChain(self.states, self.P, np.random.default_rng(source_seed))
            sender = deepcopy(self.sender)
            sender.rng = np.random.default_rng(sender_seed)
            # Give keyed senders a fresh real/fake schedule for each training run. Used by Sofia_Keyed
            if hasattr(sender, "key_rng"):
                sender.key_rng = np.random.default_rng(key_seed)
            bob_channel = Channel(self.bob_reception_probability, np.random.default_rng(bob_seed))
            eve_channel = Channel(self.eve_reception_probability, np.random.default_rng(eve_seed))
            window = ObservationWindow()
            for index in chain.sample(self.T_per_run, start_state=0):
                true_state = self.states[index]
                packet = sender.transmit(true_state)
                bob_received = bob_channel.observe(packet)
                eve_received = eve_channel.observe(packet)
                # All existing Bob decoders ACK iff a packet was received.
                # Decoding itself cannot affect the observations used by Eve.
                ack = bob_received is not None
                X.append(window.features(eve_received))
                y.append(int(true_state[1]))
                
                window.update(eve_received, ack)
                if self.feedback is not None:
                    self.feedback(sender, int(index), ack)
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

    def train(self):
        X, y = self.generate_data()
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "schema_version": SCHEMA_VERSION,
            "feature_count": FEATURE_COUNT,
            "defense": self.defense,
            "training_seed": self.seed,
            "training_samples": len(y),
            "states": self.states,
            "transition_matrix": self.P,
            "bob_reception_probability": self.bob_reception_probability,
            "eve_reception_probability": self.eve_reception_probability,
            "sender_parameters": {
                name: getattr(self.sender, name)
                for name in ("sending_probability", "theta", "Bob_window_size")
                if hasattr(self.sender, name)
            },
        }, self.model_path)
        print(f"Trained {self.defense}: {len(y)} samples, {X.shape[1]} features")
        print("Saved model to:", self.model_path)
        return self.model_path
