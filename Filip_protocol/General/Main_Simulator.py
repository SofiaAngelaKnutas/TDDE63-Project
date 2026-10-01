"""Connect Alice, Bob, and Eve and report simulation metrics.

Run from the repository root with:
    python -m Filip_protocol.General.Main_Simulator
or:
    python Filip_protocol/General/Main_Simulator.py
"""

from pathlib import Path
import sys
import time

import numpy as np

# Support direct-file execution as well as ``python -m`` from the repository
# root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Filip_protocol.General.Alice import Alice
from Filip_protocol.General.Bob import Bob
from Filip_protocol.General.Eve import Eve
from Filip_protocol.General.markov_model import STATES, TRANSITION_MATRIX


class Main_Simulator:
    """Coordinate packet delivery, collect estimates, and calculate metrics."""

    def __init__(self, sequence_length=1_000_000):
        self.sequence_length = sequence_length
        initial_state_index = 0
        initial_state = STATES[initial_state_index]

        self.alice = Alice(
            states=STATES,
            transition_matrix=TRANSITION_MATRIX,
            sequence_length=sequence_length,
            sending_probability=0.93,
            start_state_index=initial_state_index,
            source_rng=np.random.default_rng(42),
            sender_rng=np.random.default_rng(43),
        )
        self.bob = Bob(
            states=STATES,
            transition_matrix=TRANSITION_MATRIX,
            packet_loss=0.05,
            rng=np.random.default_rng(44),
            initial_estimate=initial_state,
        )
        self.eve = Eve(
            packet_loss=0.25,
            rng=np.random.default_rng(45),
            initial_estimate=initial_state,
        )

    def run(self):
        start_time = time.perf_counter()
        bob_predictions = np.empty_like(self.alice.sequence)
        eve_predictions = np.empty_like(self.alice.sequence)
        # The initial state is shared; receivers predict only later time steps.
        initial_state = self.alice.sequence[0]
        bob_predictions[0] = initial_state
        eve_predictions[0] = initial_state
        progress_interval = max(1, self.sequence_length // 10)

        for step in range(1, self.sequence_length):
            packet = self.alice.send_packet(step)
            bob_predictions[step] = self.bob.receive_packet(packet)
            eve_predictions[step] = self.eve.receive_packet(packet)

            if (step + 1) % progress_interval == 0 or step + 1 == self.sequence_length:
                progress = (step + 1) / self.sequence_length * 100
                print(f"\rProgress: {progress:.0f}%", end="", flush=True)
        print()

        true_states = self.alice.sequence
        bob_y = bob_predictions[:, 0]
        bob_s = bob_predictions[:, 1]
        eve_s = eve_predictions[:, 1]
        true_y = true_states[:, 0]
        true_s = true_states[:, 1]

        bob_joint_correct = (bob_y == true_y) & (bob_s == true_s)
        eve_s_correct = eve_s == true_s
        cra = bob_joint_correct & ~eve_s_correct

        print(f"Simulation time: {time.perf_counter() - start_time:.2f} seconds")
        print(f"Sequence length (T): {self.sequence_length:,}")
        print(f"Packets sent by Alice: {self.alice.packets_sent:,}")
        print(f"Bob packets lost: {self.bob.packets_lost:,}")
        print(f"Eve packets lost: {self.eve.packets_lost:,}")
        print(f"Bob public Y accuracy: {np.mean(bob_y == true_y):.4f}")
        print(f"Bob private S accuracy: {np.mean(bob_s == true_s):.4f}")
        print(f"Bob joint (Y + S) accuracy: {np.mean(bob_joint_correct):.4f}")
        print(f"Eve S accuracy: {np.mean(eve_s_correct):.4f}")
        print(f"Average CRA: {np.mean(cra):.4f}")


if __name__ == "__main__":
    Main_Simulator().run()
