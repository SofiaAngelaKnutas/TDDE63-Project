import numpy as np
import math #we need e log and ^

class Channel:

    def __init__(self, packet_loss=0.0, rng=None):
        self.packet_loss = packet_loss
        self.rng = rng or np.random.default_rng()

    def observe(self, packet):
        if self.rng.random() < self.packet_loss:
            return None

        return packet

class JammingChannel:
    """
    Receiver jamming defense (Zheng et al., "Improving Physical Layer Secrecy
    Using Full-Duplex Jamming Receivers", IEEE TSP 2013).

    Bob sends a jamming signal while he receives Alice's packets. We do not
    simulate the signal itself; we calculate the chance that a packet gets through.
    """

    def __init__(self, reception_probability, jam_power, rng=None):
        self.threshold = 1.0 # Minimum SNR ratio needed to decode a packet.
        #Set to 1 to make math easier.
        
        # Signal strength at, chosen so that WITHOUT jamming the packet loss
        # is the same as in simulator file.
        # Eve is further away from Alice, so her signal is weaker.
        # Basically; How well Bob or Eve hears Alice's signal without jamming. Bob is closer, so he hears better.
        # SNR is how loud Alice is at each receiver. Bob is close so it sounds louder to him than to eve 
        self.SNR = -self.threshold / math.log(reception_probability)   
        # Example:
        # Bob = 19.5 - For Bob to lose 5 %, the signal at him must be 19.5.
        # Eve = 3.48 - For Eve to lose 25 %, the signal at her must be 3.48. She is "further away" so she hears worse than Bob.   
        
        self.rng = rng or np.random.default_rng()
        
        """
        Chance that a packet gets through (Rayleigh fading).
        Both Alice's signal and the jamming fade randomly (exponential variables).
        Averaging over that randomness gives:  e^(-T/SNR) / (1 + T * jam / SNR)
            first part:  chance the signal beats the normal background noise
            second part: how much the jamming lowers that chance
        """
        self.p_jam = math.exp(-self.threshold / self.SNR) / (1 + self.threshold * jam_power / self.SNR)

        self.packet_loss = 1 - self.p_jam
    
    def observe(self, packet):
        if self.rng.random() < self.packet_loss:
            return None

        return packet

    def get_packet_loss(self):
        return self.packet_loss
