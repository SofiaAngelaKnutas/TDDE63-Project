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


def run(jam_power=0.0, jam_fraction=1.0, eve_receiver="mrc", eve_antennas=2,
        eve_learns_from="true_states", T=5000, seed=42):
    """
    eve_learns_from:
      "true_states"   - as in main: true states become public after each step,
                        so Eve's P' does not depend on what she received
      "own_estimates" - Eve only learns from her own decoded states
    """
    rng = np.random.default_rng(seed)
    chain = MarkovChain(states, P, rng=np.random.default_rng(seed))
    encoder = Encoder()

    # ---- DEFENSE: one jammer shared by both channels ----
    jammer = Jammer(power=jam_power, jam_fraction=jam_fraction, rng=rng)
    bob_channel = JammingChannel("bob", jammer, noise_std=0.4, rng=rng)
    eve_channel = JammingChannel("eve", jammer, n_antennas=eve_antennas,
                                 eve_receiver=eve_receiver, noise_std=0.8, rng=rng)
    # -----------------------------------------------------

    trajectory = chain.sample(T=T, start_state=0)
    bob_decoder = Decoder(P, states, encoder, bob_channel.noise_std)
    eve_learner = TransitionLearner(n_states=len(states), alpha=1.0)
    eve_decoder = Decoder(eve_learner.P, states, encoder, eve_channel.noise_std)

    bob_pred, eve_pred = [], []
    for t in range(T):
        jammer.step()  # Bob decides whether he jams this timestep

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


if __name__ == "__main__":
    print(f"{'Eve learns from':<15} {'Eve rx':<5} {'jam':>5} | {'Bob acc':>7} {'Eve acc':>7} {'||P-P_eve||':>11}")
    print("-" * 62)
    for learns in ("true_states", "own_estimates"):
        for receiver in ("mrc", "mmse"):
            for pd in (0.0, 10.0, 100.0):
                r = run(jam_power=pd, eve_receiver=receiver, eve_learns_from=learns)
                print(f"{learns:<15} {receiver:<5} {pd:>5} | "
                      f"{r['bob_acc']:>7.3f} {r['eve_acc']:>7.3f} {r['P_error']:>11.3f}")
