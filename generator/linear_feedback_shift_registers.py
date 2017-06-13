from functools import reduce
from operator import xor

from generator.generator import BitGenerator


class LinearFeedbackShiftRegister(BitGenerator):
    """
    Linear feedback shift register to generate random bits, depending on the given tap sequence
    and initial register state (seed).
    """

    NAME = "Linear Feedback Shift Register"

    def info(self):
        return [self.NAME,
                "parameters: length=%d, taps=%s" % (self.length, str(self.taps)),
                "seed (register state): " + str(self.state())]

    def state(self):
        return self.register

    def __init__(self, length, taps, seed=0):
        """
        """

        self.length = length
        self.taps = taps
        self.register = seed

        super().__init__(seed)

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator.
        """
        super().seed(a, version)

        self.register = a

    def random_bit(self):
        last_bit = 1 & self.register

        # tap on the bits
        new_bit = 1 & reduce(xor, [self.register >> (p - 1) for p in self.taps])

        self.register = (new_bit << (self.length - 1)) | (self.register >> 1)

        return last_bit

    def getstate(self):
        return self.VERSION, self.length, self.taps, self.register

    def setstate(self, state):
        self.VERSION = state[0]
        self.length = state[1]
        self.taps = state[2]
        self.register = state[3]
