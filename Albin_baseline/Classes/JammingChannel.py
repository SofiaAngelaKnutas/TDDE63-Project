"""
Receiver jamming (idea from Zheng et al., "Improving Physical Layer Secrecy
Using Full-Duplex Jamming Receivers", IEEE TSP 2013).

Bob sends noise while he receives Alice's packets:
  1. Encoder:  each bit of the state [Y, S] becomes a number: 0 -> -1.0, 1 -> +1.0
  2. Channel:  the jamming noise is added to those numbers
  3. Decoder:  each number is turned back into a bit: positive -> 1, negative -> 0
If the noise pushes a number across 0, that bit flips and the receiver gets the
wrong state. Eve gets the full jamming noise. Bob gets only the part that leaks
into his own receiver, because he knows the noise he sends and removes most of it.
"""
import math

import numpy as np

from Classes.Channel import Channel


def encode(state):
    """Bits -> numbers: 0 -> -1.0, 1 -> +1.0"""
    return 2.0 * np.asarray(state, dtype=float) - 1.0


def decode(signal):
    """Numbers -> bits: positive -> 1, negative -> 0"""
    return (np.asarray(signal) > 0).astype(int)


class JammingChannel(Channel):
    """
    Albin's Channel (packets lost with probability base_loss) plus jamming noise
    on every packet that arrives.

    jam_noise: how strong Bob's jamming noise is (standard deviation, 0 = no jamming)
    leak:      share of the noise power that reaches this receiver
               (Eve: 1.0, Bob: small, e.g. 0.1)
    """

    def __init__(self, base_loss, jam_noise, leak=1.0, rng=None):
        super().__init__(packet_loss=base_loss, rng=rng)
        self.noise_std = math.sqrt(leak) * jam_noise    # noise this receiver actually gets

    @property
    def bit_error_prob(self):
        """Chance that the noise flips one bit (the noise pushes the number across 0)."""
        if self.noise_std == 0:
            return 0.0
        return 0.5 * math.erfc(1 / (self.noise_std * math.sqrt(2)))

    def observe(self, packet):
        packet = super().observe(packet)                   # normal packet loss first
        if packet is None:
            return None
        signal = encode(packet) + self.rng.normal(0.0, self.noise_std, size=len(packet))
        return decode(signal)
