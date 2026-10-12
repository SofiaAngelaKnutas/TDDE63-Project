"""
Receiver jamming defense (Zheng et al., "Improving Physical Layer Secrecy
Using Full-Duplex Jamming Receivers", IEEE TSP 2013).

Bob sends a jamming signal while he receives Alice's packets. We do not
simulate the signal itself; we calculate the chance that a packet gets through.
"""
import math #we need e log and ^

THRESHOLD = 1.0   # minimum signal-to-noise ratio needed to decode a packet
#why this? why exactly 1? because it makes the math easier, and the actual value doesn't matter

# Signal strength at Bob and Eve, chosen so that WITHOUT jamming the packet loss
# is the same as in Simulator_Simple.py (Bob 5 %, Eve 25 %).
# Eve is further away from Alice, so her signal is weaker.
#simple; how well Bob and Eve hear Alice's signal without jamming. Bob is closer, so he hears better.
#SNR is how loud Alice is at each receiver. Bob is close so it sounds louder to him than to eve 
#baseline needs to match the group, therefore the SNRs are chosen so that the packet loss is the same as in Simulator_Simple.py (Bob 5 %, Eve 25 %).
SNR_BOB = -THRESHOLD / math.log(1 - 0.05)   # = 19.5 - For Bob to lose 5 %, the signal at him must be 19.5.
SNR_EVE = -THRESHOLD / math.log(1 - 0.25)   # = 3.48 - For Eve to lose 25 %, the signal at her must be 3.48. She is "further away" so she hears worse than Bob.


#the chance that a packet gets through (Rayleigh fading).
def p_success(snr, jam):
    """
    Chance that a packet gets through (Rayleigh fading).
    Both Alice's signal and the jamming fade randomly (exponential variables).
    Averaging over that randomness gives:  e^(-T/SNR) / (1 + T * jam / SNR)
      first part:  chance the signal beats the normal background noise
      second part: how much the jamming lowers that chance
    """
    return math.exp(-THRESHOLD / snr) / (1 + THRESHOLD * jam / snr)
#math.exp(-THRESHOLD / snr) is the chance that the signal beats the channel. Loud Alice gives a better chance. 
#(1 + THRESHOLD * jam / snr) is how much the jamming lowers that chance.
# More radio (jam) means a bigger number underneath and a smaller chance. And if Alice is loud (big snr), the radio matters less.
#“expectations of exponential variables”
#The shouts and the radio randomly get louder and quieter; the formula is the average over all of that.


#defense

def jamming_probabilities(jam_power, self_interference):
    """
    The chance that a packet reaches Bob (P_AB) and Eve (P_AE) when Bob jams.
    Eve gets all of the jamming. Bob only gets the part of his own jamming he
    cannot cancel (self_interference, 0 = perfect cancellation).
    """
    p_ab = p_success(SNR_BOB, self_interference * jam_power) #Bob: loud Alice, but he hears some of his own jamming.
    p_ae = p_success(SNR_EVE, jam_power) #Eve: quieter Alice, but she hears all of Bob's jamming.
    return p_ab, p_ae #chance Bob hears it and chance Eve hears it

