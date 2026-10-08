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


# Eve's hardware is the same in every simulation. Two antennas are needed for the
# aware (MMSE) Eve to be able to suppress jamming at all: with one antenna, MMSE
# and MRC give exactly the same SINR.
EVE_ANTENNAS = 2
EVE_RECEIVERS = ("mrc", "mmse")


class JammingChannel:
    """
    Drop-in replacement for Channel. Instead of a fixed packet_loss, a packet
    is lost when the SINR falls below `threshold`.

    role="bob": SINR from paper eq. 3 (self-interference scaled by rho)
    role="eve": EVE_ANTENNAS antennas (fixed), with one of two receivers:
                "mrc"  - unaware of the jamming, treats it as noise (paper eq. 3)
                "mmse" - knows about the jamming and suppresses it (paper eq. 41)
    """

    def __init__(self, role, jammer, Ps=10.0, threshold=1.0, rho=0.05,
                 eve_receiver="mrc", noise_std=0.0, rng=None):
        if role not in ("bob", "eve"):
            raise ValueError("role must be 'bob' or 'eve'")
        if eve_receiver not in EVE_RECEIVERS:
            raise ValueError(f"eve_receiver must be one of {EVE_RECEIVERS}")
        self.role = role
        self.eve_receiver = eve_receiver
        self.jammer = jammer
        self.Ps = Ps
        self.threshold = threshold
        self.rho = rho
        self.noise_std = noise_std  # kept so Decoder(observation_noise_std=...) still works
        self.rng = rng or np.random.default_rng()

    def sinr(self):
        j = self.jammer
        if self.role == "bob":
            self_interference = np.real(j.r.conj() @ j.H_loop @ j.Q @ j.H_loop.conj().T @ j.r)
            return self.Ps * np.linalg.norm(j.h_sd) ** 2 / (1 + self.rho * self_interference)

        h_se = _cn(self.rng, EVE_ANTENNAS)                    # Alice -> Eve
        H_ed = _cn(self.rng, EVE_ANTENNAS, j.Mt)              # Bob's jammer -> Eve
        J = H_ed @ j.Q @ H_ed.conj().T                        # jamming Eve receives
        if self.eve_receiver == "mmse":                       # aware Eve suppresses the jamming
            return self.Ps * np.real(h_se.conj() @ np.linalg.solve(J + np.eye(EVE_ANTENNAS), h_se))
        g = np.linalg.norm(h_se) ** 2                         # unaware Eve: MRC, jamming = noise
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
