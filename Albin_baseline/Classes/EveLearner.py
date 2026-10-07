from collections import deque

import joblib
import numpy as np


class ObservationWindow:
    def __init__(self, window_size=3, observation_size=2):
        self.window_size = window_size
        self.observation_size = observation_size

        # Contains previous observations only.
        self.history = deque(
            [None] * (window_size - 1),
            maxlen=window_size - 1,
        )

    #Sometimes we send "None" packets. Random Forest cannot interpret "None", so we transform to [-1, -1]
    def _encode(self, observation):
        if observation is None:
            return np.full(self.observation_size, -1.0)

        observation = np.asarray(observation, dtype=float)

        if observation.size != self.observation_size:
            raise ValueError(
                f"Expected observation size {self.observation_size}, "
                f"got {observation.size}"
            )

        return observation

    def features(self, current_observation):
        observations = list(self.history) + [current_observation]

        return np.concatenate([
            self._encode(obs)
            for obs in observations
        ])

    def update(self, observation):
        if self.window_size > 1:
            self.history.append(observation)


class RandomForestEve:
    def __init__(self, model_path):
        saved = joblib.load(model_path)

        self.model = saved["model"]

        self.window = ObservationWindow( #We dont need to load this, but could possibly prevent some accident :)
            window_size=saved["window_size"],
            observation_size=saved["observation_size"],
        )

    def observe(self, observation):
        """
        Build feature row from current observation,
        then update Eve's observation history.
        """
        features = self.window.features(observation)

        self.window.update(observation)

        return features

    def predict(self, features):
        features = np.asarray(features, dtype=float)

        return self.model.predict(features).astype(int)