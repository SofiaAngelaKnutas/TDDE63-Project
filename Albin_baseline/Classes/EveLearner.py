from collections import deque

import joblib
import numpy as np

SCHEMA_VERSION = "packets2-history5-ack-v1"
FEATURE_COUNT = 27


class ObservationWindow:
    """Five previous slots, plus the last successful packet/feedback.

    Freeze features(packet) before update(packet, ack). Frozen rows can be
    predicted together after the run. Neither operation accepts hidden state.
    Packet encoding: [present, value0, value1]. Feedback: None=-1, NACK=0,
    ACK=1. Startup padding uses the same encoding as missing observations.
    """

    def __init__(self, window_size=5, observation_size=2):
        if window_size != 5 or observation_size != 2:
            raise ValueError("Unified Eve requires five previous slots and two-value packets")
        self.history = deque([None] * 5, maxlen=5)
        self.ack_history = deque([None] * 5, maxlen=5)
        self.last_received_packet = None
        self.last_received_ack = None

    @staticmethod
    def _encode(packet):
        if packet is None:
            return np.zeros(3)
        values = np.asarray(packet, dtype=float).reshape(-1)
        if values.size != 2 or not np.all(np.isfinite(values)):
            raise ValueError("Expected a finite two-value packet or None")
        return np.concatenate(([1.0], values))

    @staticmethod
    def _encode_ack(ack):
        if ack is None:
            return -1.0
        if ack not in (0, 1):
            raise ValueError("ACK/NACK must be True/1, False/0, or None")
        return float(ack)

    def features(self, current_observation):
        """Return a fresh row without changing history.

        [current packet (3), last received packet (3), last feedback (1),
        five previous packets (15), five previous feedback values (5)].
        Both histories are ordered oldest to newest.
        """
        return np.concatenate([
            self._encode(current_observation),
            self._encode(self.last_received_packet),
            [self._encode_ack(self.last_received_ack)],
            *(self._encode(packet) for packet in self.history),
            [self._encode_ack(ack) for ack in self.ack_history],
        ])

    def update(self, observation, ack):
        """Commit this slot after its features are frozen"""
        self._encode(observation)
        self._encode_ack(ack)
        packet = None if observation is None else np.asarray(observation).reshape(-1).copy()
        self.history.append(packet)
        self.ack_history.append(ack)
        if packet is not None:
            self.last_received_packet = packet
        if ack is not None:
            self.last_received_ack = ack


class RandomForestEve:
    def __init__(self, model_path, defense):
        saved = joblib.load(model_path)
        if saved.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Incompatible Eve feature schema; retrain this model")
        if saved.get("defense") != defense:
            raise ValueError(f"Expected {defense!r} model, got {saved.get('defense')!r}")
        self.model = saved["model"]
        if saved.get("feature_count") != FEATURE_COUNT or self.model.n_features_in_ != FEATURE_COUNT:
            raise ValueError("Incompatible Eve feature dimensions; retrain this model")
        self.window = ObservationWindow()

    def observe(self, observation):
        """Build features without committing the slot or receiving current ACK."""
        return self.window.features(observation)

    def predict(self, features):
        rows = np.asarray(features, dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != FEATURE_COUNT:
            raise ValueError(f"Expected feature rows with {FEATURE_COUNT} columns")
        return self.model.predict(rows).astype(int)

    def update(self, observation, ack):
        self.window.update(observation, ack)
