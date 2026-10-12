from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Classes.MarkovChain import MarkovChain
from Classes.Channel import Channel
from Classes.EveLearner import ObservationWindow, ObservationWindowACK


class RandomForest_Trainer:
    def __init__(
        self,
        model_name,
        states,
        P,
        training_runs,
        T_per_run,
        window_size,
        observation_size,
        eve_packet_loss,
        sender,
    ):
        self.model_name = model_name

        self.states = np.asarray(states)
        self.P = np.asarray(P)

        self.training_runs = training_runs
        self.T_per_run = T_per_run
        self.window_size = window_size
        self.observation_size = observation_size
        self.eve_packet_loss = eve_packet_loss

        self.sender = sender

    def train(self):
        X_train = []
        y_train = []

        master_rng = np.random.default_rng(1000)

        for run in range(self.training_runs):
            chain = MarkovChain(
                states=self.states,
                transition_matrix=self.P,
                rng=np.random.default_rng(
                    master_rng.integers(0, 2**32)
                ),
            )

            eve_channel = Channel(
                packet_loss=self.eve_packet_loss,
                rng=np.random.default_rng(
                    master_rng.integers(0, 2**32)
                ),
            )

            window = ObservationWindow(
                window_size=self.window_size,
                observation_size=self.observation_size,
            )

            state_history = chain.sample(
                T=self.T_per_run,
                start_state=0,
            )

            for t in range(self.T_per_run):
                true_state = self.states[state_history[t]]
                true_y, true_s = true_state

                sent_packet = self.sender.transmit(true_state)
                eve_received = eve_channel.observe(sent_packet)

                features = window.features(eve_received)

                X_train.append(features)

                # Ground truth is ONLY the supervised-learning label.
                y_train.append(int(true_s))

                # Eve remembers what it actually observed.
                window.update(eve_received)

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)

        print("Training examples:", X_train.shape[0])
        print("Features per example:", X_train.shape[1])

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        output_dir = Path("TempModels")
        output_dir.mkdir(exist_ok=True)

        model_path = output_dir / self.model_name

        joblib.dump(
            {
                "model": model,
                "window_size": self.window_size,
                "observation_size": self.observation_size,
                "attacker_model": "observation_window",
            },
            model_path,
        )

        print("Saved model to:", model_path)
        print("Training accuracy:", model.score(X_train, y_train))


class RandomForest_Trainer_Albin:
    def __init__(
        self,
        model_name,
        states,
        P,
        training_runs,
        T_per_run,
        window_size,
        observation_size,
        eve_packet_loss,
        sender,
        bob_packet_loss,
    ):
        self.model_name = model_name

        self.states = np.asarray(states)
        self.P = np.asarray(P)

        self.training_runs = training_runs
        self.T_per_run = T_per_run
        self.window_size = window_size
        self.observation_size = observation_size
        self.eve_packet_loss = eve_packet_loss

        self.sender = sender
        self.bob_packet_loss = bob_packet_loss

        self.state_to_idx = { #Just to translate [1, 1] == 3 for example
            tuple(state): i
            for i, state in enumerate(self.states)
        }

    def train(self):
        X_train = []
        y_train = []

        master_rng = np.random.default_rng(1000)

        for run in range(self.training_runs):
            chain = MarkovChain(
                states=self.states,
                transition_matrix=self.P,
                rng=np.random.default_rng(
                    master_rng.integers(0, 2**32)
                ),
            )

            bob_channel = Channel(
                packet_loss=self.bob_packet_loss,
                rng=np.random.default_rng(
                    master_rng.integers(0, 2**32)
                ),
            )

            eve_channel = Channel(
                packet_loss=self.eve_packet_loss,
                rng=np.random.default_rng(
                    master_rng.integers(0, 2**32)
                ),
            )

            window = ObservationWindowACK(
                window_size=self.window_size,
                observation_size=self.observation_size,
            )

            state_history = chain.sample(
                T=self.T_per_run,
                start_state=0,
            )

            self.sender.reset()

            for t in range(self.T_per_run):
                true_state = self.states[state_history[t]]
                true_y, true_s = true_state

                sent_packet = self.sender.transmit(true_state)

                bob_received = bob_channel.observe(sent_packet)
                eve_received = eve_channel.observe(sent_packet)

                ACK = bob_received is not None
                features = window.features(eve_received) #First build features

                X_train.append(features)
                y_train.append(int(true_s))

                # Slot has now ended. ACK_t becomes public.
                window.update(eve_received, ACK)

                # Alice also learns Bob's ACK/NACK.
                if ACK:
                    self.sender.update_Bob_window(
                        self.state_to_idx[tuple(true_state)]
                    )
                else:
                    self.sender.update_Bob_window(0)

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)

        print("Training examples:", X_train.shape[0])
        print("Features per example:", X_train.shape[1])

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        output_dir = Path("TempModels")
        output_dir.mkdir(exist_ok=True)

        model_path = output_dir / self.model_name

        joblib.dump(
            {
                "model": model,
                "window_size": self.window_size,
                "observation_size": self.observation_size,
                "attacker_model": "observation_window",
            },
            model_path,
        )

        print("Saved model to:", model_path)
        print("Training accuracy:", model.score(X_train, y_train))