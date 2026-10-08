"""
Jamming defense simulation.

Run with defaults:
    python Simulator_jamming.py
Choose settings:
    python Simulator_jamming.py --bob-tx-antennas 3 --strategy nullspace
Aware Eve (knows about the jamming and suppresses it):
    python Simulator_jamming.py --eve-receiver mmse

Eve always has 2 antennas. By default she is unaware of the jamming (MRC).
CRA = Bob gets the whole state right AND Eve gets the private attribute S wrong
(same definition as Albin_baseline, from Li and Pappas).
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
        strategy="isotropic", eve_receiver="mrc", eve_learns_from="true_states", Ps=10.0, rho=0.05, threshold=1.0,
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
                                 eve_receiver=eve_receiver, noise_std=0.8, rng=rng)
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

    bob_pred, eve_pred = np.array(bob_pred), np.array(eve_pred)
    bob_correct = bob_pred == trajectory                            # whole state (Y and S)
    eve_wrong_S = states[eve_pred][:, 1] != states[trajectory][:, 1]  # private attribute S
    return {
        "bob_acc": np.mean(bob_correct),
        "eve_acc": np.mean(eve_pred == trajectory),
        "eve_S_acc": np.mean(~eve_wrong_S),
        "CRA": np.mean(bob_correct & eve_wrong_S),
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
    p.add_argument("--eve-receiver", choices=["mrc", "mmse"], default="mrc",
                   help="mrc = Eve unaware of jamming, mmse = Eve knows and suppresses it")
    p.add_argument("--seeds", type=int, default=1, help="number of random seeds to average over")
    p.add_argument("--eve-learns-from", choices=["true_states", "own_estimates"],
                   default="true_states", help="true_states = same Eve as in main")
    p.add_argument("--T", type=int, default=5000, help="timesteps per run")
    a = p.parse_args()

    aware = "aware of jamming (MMSE)" if a.eve_receiver == "mmse" else "unaware of jamming (MRC)"
    print(f"Eve: 2 antennas, {aware}, learns from {a.eve_learns_from}")
    print(f"Bob: jam antennas={a.bob_tx_antennas}, rx antennas={a.bob_rx_antennas}, "
          f"strategy={a.strategy}, jam_fraction={a.jam_fraction}, Ps={a.Ps}, rho={a.rho}, "
          f"T={a.T}, seeds={a.seeds}\n")
    keys = ["bob_acc", "eve_acc", "eve_S_acc", "CRA", "P_error"]
    print(f"{'jam':>6} | {'Bob acc':>7} {'Eve acc':>7} {'Eve S acc':>9} {'CRA':>6} {'||P-P_eve||':>11}")
    print("-" * 56)
    for pd in a.jam_powers:
        runs = [run(jam_power=pd, jam_fraction=a.jam_fraction,
                    bob_tx_antennas=a.bob_tx_antennas, bob_rx_antennas=a.bob_rx_antennas,
                    strategy=a.strategy, eve_receiver=a.eve_receiver,
                    eve_learns_from=a.eve_learns_from, Ps=a.Ps, rho=a.rho,
                    threshold=a.threshold, T=a.T, seed=42 + i) for i in range(a.seeds)]
        m = {k: np.mean([r[k] for r in runs]) for k in keys}
        print(f"{pd:>6g} | {m['bob_acc']:>7.3f} {m['eve_acc']:>7.3f} {m['eve_S_acc']:>9.3f} "
              f"{m['CRA']:>6.3f} {m['P_error']:>11.3f}")

if __name__ == "__main__":
    main()
