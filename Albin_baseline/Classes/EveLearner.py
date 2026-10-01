from collections import deque
import joblib
import numpy as np


class ObservationWindow:
    """
    Builds Eve's input without leaking the current public Y or hidden S.

    Each finalized historical slot is represented as:
        [public_Y, received_flag, *raw_eve_observation]

    For the current slot, public_Y is -1 because Y is only revealed
    after Eve has guessed. If Eve receives nothing, payload values are 0;
    received_flag tells the model that those zeros are placeholders.

    Training labels (true S) are NEVER stored in this history unless S was
    genuinely part of the raw observation Eve intercepted.
    """

    def __init__(self, window_size=5, observation_size=2):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if observation_size < 1:
            raise ValueError("observation_size must be at least 1")

        self.window_size = window_size
        self.observation_size = observation_size
        self.history = deque(maxlen=window_size - 1)

    def _observation_parts(self, observation):
        if observation is None:
            return 0.0, np.zeros(self.observation_size, dtype=float)

        payload = np.asarray(observation, dtype=float).reshape(-1)
        if payload.size != self.observation_size:
            raise ValueError(
                f"Expected observation of size {self.observation_size}, "
                f"got {payload.size}"
            )

        return 1.0, payload

    def _record(self, public_y, observation):
        received_flag, payload = self._observation_parts(observation)
        return np.concatenate((
            np.array([float(public_y), received_flag]), #NOTE: This public_y will be set to -1.0, as we dont know it
            payload,
        ))

    def features_for_current(self, observation):
        """
        Build features available BEFORE current Y and S are revealed.
        """
        records = list(self.history)

        # Pad beginning of sequence to a fixed window length.
        padding_count = (self.window_size - 1) - len(records)
        padding_record = np.concatenate((
            np.array([-1.0, 0.0]),
            np.zeros(self.observation_size, dtype=float),
        ))

        padded_history = [padding_record.copy() for _ in range(padding_count)]
        padded_history.extend(records)

        # Current public Y is deliberately hidden until after the guess.
        current_record = self._record(public_y=-1.0, observation=observation)

        all_records = padded_history + [current_record]
        return np.concatenate(all_records)

    def finish_slot(self, revealed_y, observation):
        """
        Call AFTER Eve's guess, when Y becomes public.

        true S is intentionally not accepted here. It may be a training label,
        but it must not become future evaluation input unless Eve actually
        intercepted it inside `observation`.
        """
        if self.window_size == 1:
            return

        self.history.append(
            self._record(public_y=revealed_y, observation=observation)
        )


class RandomForestEve:
    """Frozen Random-Forest attacker used during evaluation."""

    def __init__(self, model_path):
        saved = joblib.load(model_path)

        self.model = saved["model"]
        self.window = ObservationWindow(
            window_size=saved["window_size"],
            observation_size=saved["observation_size"],
        )

    def features_for_current(self, observation):
        return self.window.features_for_current(observation)

    def predict_features(self, features):
        """Predict one feature vector or a whole matrix of feature vectors."""
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.model.predict(features).astype(int)

    def reveal_y(self, revealed_y, observation):
        self.window.finish_slot(revealed_y, observation)
