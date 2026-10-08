"""
Jamming defense simulation.

Run with defaults:
    python Simulator_jamming.py
Choose settings:
    python Simulator_jamming.py --bob-tx-antennas 3 --strategy nullspace

Eve is fixed (1 antenna, unaware of the jamming), so only Bob's defense varies.
See all options:
    python Simulator_jamming.py --help
"""
import argparse
import numpy as np
from Classes.MarkovChain import MarkovChain
from Classes.Encoder import Encoder
from Classes.Decoder import Decoder
from Classes.TransitionLearner import TransitionLearner
from Classes.JammingChannel import Jammer, JammingChannel

states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

P = np.array([
    [0.60, 0.15, 0.10, 0.15],
    [0.20, 0.50, 0.20, 0.10],
    [0.10, 0.20, 0.50, 0.20],
    [0.15, 0.10, 0.15, 0.60],
])


def run(jam_power=0.0, jam_fraction=1.0, bob_tx_antennas=1, bob_rx_antennas=1,
        strategy="isotropic", eve_learns_from="true_states", Ps=10.0, rho=0.05, threshold=1.0,
        T=5000, seed=42):
    """
    eve_learns_from:
      "true_states"   - as in main: true states become public after each step,
                        so Eve's P' does not depend on what she received
      "own_estimates" - Eve only learns from her own decoded states
    """
    rng = np.random.default_rng(seed)
    chain = MarkovChain(states, P, rng=np.random.default_rng(seed))
    encoder = Encoder()

    # ---- DEFENSE: Bob's full-duplex jammer, shared by both channels ----
    jammer = Jammer(power=jam_power, jam_fraction=jam_fraction,
                    tx_antennas=bob_tx_antennas, rx_antennas=bob_rx_antennas,
                    strategy=strategy, rng=rng)
    bob_channel = JammingChannel("bob", jammer, Ps=Ps, threshold=threshold,
                                 rho=rho, noise_std=0.4, rng=rng)
    eve_channel = JammingChannel("eve", jammer, Ps=Ps, threshold=threshold,
                                 noise_std=0.8, rng=rng)
    # --------------------------------------------------------------------

    trajectory = chain.sample(T=T, start_state=0)
    bob_decoder = Decoder(P, states, encoder, bob_channel.noise_std)
    eve_learner = TransitionLearner(n_states=len(states), alpha=1.0)
    eve_decoder = Decoder(eve_learner.P, states, encoder, eve_channel.noise_std)

    bob_pred, eve_pred = [], []
    for t in range(T):
        jammer.step()  # Bob decides whether he jams + new channel realisation

        encoded = encoder.encode(states[trajectory[t]])
        bob_received = bob_channel.transmit(encoded)
        eve_received = eve_channel.transmit(encoded)

        eve_decoder.set_transition_matrix(eve_learner.P)
        if t == 0:
            bob_pred.append(bob_decoder.observe_initial(bob_received))
            eve_pred.append(eve_decoder.observe_initial(eve_received))
        else:
            bob_pred.append(bob_decoder.step(bob_received))
            eve_pred.append(eve_decoder.step(eve_received))
            if eve_learns_from == "true_states":
                eve_learner.update(trajectory[t - 1], trajectory[t])
            else:
                eve_learner.update(eve_pred[t - 1], eve_pred[t])

    return {
        "bob_acc": np.mean(np.array(bob_pred) == trajectory),
        "eve_acc": np.mean(np.array(eve_pred) == trajectory),
        "P_error": np.linalg.norm(P - eve_learner.P, ord="fro"),
    }


def main():
    p = argparse.ArgumentParser(description="Receiver jamming defense simulation")
    p.add_argument("--bob-tx-antennas", type=int, default=1, help="Bob's jamming antennas (Mt)")
    p.add_argument("--bob-rx-antennas", type=int, default=1, help="Bob's receive antennas (Mr)")
    p.add_argument("--strategy", choices=["isotropic", "nullspace"], default="isotropic",
                   help="how Bob spreads the jamming (nullspace needs Mt >= 2)")
    p.add_argument("--jam-powers", type=float, nargs="+", default=[0.0, 10.0, 100.0],
                   help="jamming powers to compare")
    p.add_argument("--jam-fraction", type=float, default=1.0, help="share of timesteps Bob jams")
    p.add_argument("--Ps", type=float, default=10.0, help="Alice's transmit power")
    p.add_argument("--rho", type=float, default=0.05, help="self-interference level at Bob (0-1)")
    p.add_argument("--threshold", type=float, default=1.0, help="min SINR to decode a packet")
    p.add_argument("--eve-learns-from", choices=["true_states", "own_estimates"],
                   default="true_states", help="true_states = same Eve as in main")
    p.add_argument("--T", type=int, default=5000, help="timesteps per run")
    a = p.parse_args()

    print(f"Eve: 1 antenna, unaware of jamming, learns from {a.eve_learns_from}")
    print(f"Bob: jam antennas={a.bob_tx_antennas}, rx antennas={a.bob_rx_antennas}, "
          f"strategy={a.strategy}, jam_fraction={a.jam_fraction}, Ps={a.Ps}, rho={a.rho}, T={a.T}\n")
    print(f"{'jam':>6} | {'Bob acc':>7} {'Eve acc':>7} {'||P-P_eve||':>11}")
    print("-" * 38)
    for pd in a.jam_powers:
        r = run(jam_power=pd, jam_fraction=a.jam_fraction,
                bob_tx_antennas=a.bob_tx_antennas, bob_rx_antennas=a.bob_rx_antennas,
                strategy=a.strategy, eve_learns_from=a.eve_learns_from,
                Ps=a.Ps, rho=a.rho, threshold=a.threshold, T=a.T)
        print(f"{pd:>6g} | {r['bob_acc']:>7.3f} {r['eve_acc']:>7.3f} {r['P_error']:>11.3f}")


if __name__ == "__main__":
    main()
