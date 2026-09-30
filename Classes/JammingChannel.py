import numpy as np


def _cn(rng, *shape):
    """Complex Gaussian CN(0,1) entries = Rayleigh fading channel coefficients."""
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)


class Jammer:
    """
    Bob's full-duplex node (Zheng et al., "Improving Physical Layer Secrecy
    Using Full-Duplex Jamming Receivers", IEEE TSP 2013).

    Bob receives with `rx_antennas` (Mr) and jams with `tx_antennas` (Mt)
    at the same time, with total jamming power `power`.

    strategy:
      "isotropic" - jamming spread equally over all directions, Q = pd/Mt * I
      "nullspace" - jam in every direction EXCEPT the one that hits Bob's own
                    receiver (paper Sec. V, Q = pd*W*W^H/(Mt-1)). Needs Mt >= 2,
                    removes Bob's self-interference.

    Call step() once per timestep: it decides whether Bob jams, draws Bob's
    channels and sets the jamming covariance Q that both channels use.
    """

    def __init__(self, power=10.0, jam_fraction=1.0, tx_antennas=1,
                 rx_antennas=1, strategy="isotropic", rng=None):
        if strategy not in ("isotropic", "nullspace"):
            raise ValueError("strategy must be 'isotropic' or 'nullspace'")
        if strategy == "nullspace" and tx_antennas < 2:
            raise ValueError("'nullspace' needs tx_antennas >= 2")
        self.power = power
        self.jam_fraction = jam_fraction
        self.Mt = tx_antennas
        self.Mr = rx_antennas
        self.strategy = strategy
        self.rng = rng or np.random.default_rng()
        self.step()

    def step(self):
        self.active = self.rng.random() < self.jam_fraction
        self.h_sd = _cn(self.rng, self.Mr)                 # Alice -> Bob
        self.H_loop = _cn(self.rng, self.Mr, self.Mt)      # Bob's jammer -> Bob's receiver
        self.r = self.h_sd / np.linalg.norm(self.h_sd)     # Bob's MRC receiver

        if not self.active or self.power == 0:
            self.Q = np.zeros((self.Mt, self.Mt))
        elif self.strategy == "isotropic":
            self.Q = self.power / self.Mt * np.eye(self.Mt)
        else:
            a = self.H_loop.conj().T @ self.r              # direction that hits Bob
            _, _, Vh = np.linalg.svd(a.conj()[None, :])
            W = Vh[1:].conj().T                            # (Mt-1) directions that don't
            self.Q = self.power / (self.Mt - 1) * W @ W.conj().T


class JammingChannel:
    """
    Drop-in replacement for Channel. Instead of a fixed packet_loss, a packet
    is lost when the SINR falls below `threshold`.

    role="bob": SINR from paper eq. 3 (self-interference scaled by rho)
    role="eve": naive MRC receiver (eq. 3) or smart MMSE receiver (eq. 41),
                with n_antennas (Me) antennas.
    """

    def __init__(self, role, jammer, Ps=10.0, threshold=1.0, rho=0.05,
                 n_antennas=1, eve_receiver="mrc", noise_std=0.0, rng=None):
        if role not in ("bob", "eve"):
            raise ValueError("role must be 'bob' or 'eve'")
        if eve_receiver not in ("mrc", "mmse"):
            raise ValueError("eve_receiver must be 'mrc' or 'mmse'")
        self.role = role
        self.jammer = jammer
        self.Ps = Ps
        self.threshold = threshold
        self.rho = rho
        self.n_antennas = n_antennas
        self.eve_receiver = eve_receiver
        self.noise_std = noise_std  # kept so Decoder(observation_noise_std=...) still works
        self.rng = rng or np.random.default_rng()

    def sinr(self):
        j = self.jammer
        if self.role == "bob":
            self_interference = np.real(j.r.conj() @ j.H_loop @ j.Q @ j.H_loop.conj().T @ j.r)
            return self.Ps * np.linalg.norm(j.h_sd) ** 2 / (1 + self.rho * self_interference)

        h_se = _cn(self.rng, self.n_antennas)                 # Alice -> Eve
        H_ed = _cn(self.rng, self.n_antennas, j.Mt)           # Bob's jammer -> Eve
        J = H_ed @ j.Q @ H_ed.conj().T                        # jamming Eve receives
        if self.eve_receiver == "mmse":                       # Eve suppresses the jamming
            return self.Ps * np.real(h_se.conj() @ np.linalg.solve(J + np.eye(self.n_antennas), h_se))
        g = np.linalg.norm(h_se) ** 2                         # naive MRC: ignores the jamming
        return self.Ps * g / (1 + np.real(h_se.conj() @ J @ h_se) / g)

    def received(self):
        return self.sinr() >= self.threshold

    # Interface used in main (Simulator.py): adds Gaussian observation noise
    def transmit(self, encoded):
        if not self.received():
            return None
        if self.noise_std > 0:
            encoded = encoded + self.rng.normal(0.0, self.noise_std, size=encoded.shape)
        return encoded

    # Interface used in Albin_baseline: packet passes through unchanged
    def observe(self, packet):
        if packet is None or not self.received():
            return None
        return packet
