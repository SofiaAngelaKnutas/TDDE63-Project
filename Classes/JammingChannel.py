import numpy as np


class Jammer:
    """
    Bob's full-duplex jammer (Zheng et al., "Improving Physical Layer Secrecy
    Using Full-Duplex Jamming Receivers", IEEE TSP 2013).

    Bob transmits jamming noise with power `power` while he receives.
    `jam_fraction` = share of timesteps he jams (1.0 = always).
    Call step() once per timestep so Bob's and Eve's channel see the same decision.
    """

    def __init__(self, power=10.0, jam_fraction=1.0, rng=None):
        self.power = power
        self.jam_fraction = jam_fraction
        self.rng = rng or np.random.default_rng()
        self.active = False

    def step(self):
        self.active = self.rng.random() < self.jam_fraction

    @property
    def current_power(self):
        return self.power if self.active else 0.0


class JammingChannel:
    """
    Drop-in replacement for Channel. Instead of a fixed packet_loss, a packet
    is lost when the SINR falls below `threshold` (fresh Rayleigh fading
    every timestep).

    role="bob": SINR = Ps|h_sd|^2 / (1 + rho*pd*|h_loop|^2)          (paper eq. 4)
    role="eve": naive MRC receiver (eq. 3) or smart MMSE receiver (eq. 41)
                with n_antennas antennas.
    """

    def __init__(self, role, jammer, Ps=10.0, threshold=1.0, rho=0.05,
                 n_antennas=1, eve_receiver="mrc", noise_std=0.0, rng=None):
        if role not in ("bob", "eve"):
            raise ValueError("role must be 'bob' or 'eve'")
        self.role = role
        self.jammer = jammer
        self.Ps = Ps
        self.threshold = threshold
        self.rho = rho
        self.n_antennas = n_antennas
        self.eve_receiver = eve_receiver
        self.noise_std = noise_std  # kept so Decoder(observation_noise_std=...) still works
        self.rng = rng or np.random.default_rng()

    def _cn(self, n):
        """n complex Gaussian CN(0,1) channel coefficients (Rayleigh fading)."""
        return (self.rng.normal(size=n) + 1j * self.rng.normal(size=n)) / np.sqrt(2)

    def sinr(self):
        pd = self.jammer.current_power

        if self.role == "bob":
            h_sd, h_loop = self._cn(1)[0], self._cn(1)[0]
            return self.Ps * abs(h_sd) ** 2 / (1 + self.rho * pd * abs(h_loop) ** 2)

        h_se, h_ed = self._cn(self.n_antennas), self._cn(self.n_antennas)
        if self.eve_receiver == "mmse":  # Eve knows about the jamming and suppresses it
            R = pd * np.outer(h_ed, h_ed.conj()) + np.eye(self.n_antennas)
            return self.Ps * np.real(h_se.conj() @ np.linalg.solve(R, h_se))
        g = np.sum(abs(h_se) ** 2)  # naive MRC: ignores the jamming
        return self.Ps * g / (1 + pd * abs(h_se.conj() @ h_ed) ** 2 / g)

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
