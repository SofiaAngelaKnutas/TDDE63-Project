"""
Receiver jamming defense (Zheng et al., "Improving Physical Layer Secrecy
Using Full-Duplex Jamming Receivers", IEEE TSP 2013).

Bob sends a jamming signal while he receives Alice's packets. We do not
simulate the signal itself; we calculate the chance that a packet gets through.
"""
import math

THRESHOLD = 1.0   # minimum signal-to-noise ratio needed to decode a packet

# Signal strength at Bob and Eve, chosen so that WITHOUT jamming the packet loss
# is the same as in Simulator_Simple.py (Bob 5 %, Eve 25 %).
# Eve is further away from Alice, so her signal is weaker.
SNR_BOB = -THRESHOLD / math.log(1 - 0.05)   # = 19.5
SNR_EVE = -THRESHOLD / math.log(1 - 0.25)   # = 3.48


def p_success(snr, jam):
    """
    Chance that a packet gets through (Rayleigh fading).
    Both Alice's signal and the jamming fade randomly (exponential variables).
    Averaging over that randomness gives:  e^(-T/SNR) / (1 + T * jam / SNR)
      first part:  chance the signal beats the normal background noise
      second part: how much the jamming lowers that chance
    """
    return math.exp(-THRESHOLD / snr) / (1 + THRESHOLD * jam / snr)


def jamming_probabilities(jam_power, self_interference):
    """
    The chance that a packet reaches Bob (P_AB) and Eve (P_AE) when Bob jams.
    Eve gets all of the jamming. Bob only gets the part of his own jamming he
    cannot cancel (self_interference, 0 = perfect cancellation).
    """
    p_ab = p_success(SNR_BOB, self_interference * jam_power)
    p_ae = p_success(SNR_EVE, jam_power)
    return p_ab, p_ae
