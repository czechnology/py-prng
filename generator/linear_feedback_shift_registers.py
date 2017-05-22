from functools import reduce
from operator import xor

from generator.generator import BitGenerator
from utils.bit_tools import least_significant_bit as lsb


class LinearFeedbackShiftRegister(BitGenerator):
    # TODO
    """ """

    NAME = "Linear Feedback Shift Register"

    def info(self):
        return [self.NAME,
                "parameters: length=%d, taps=%s" % (self.length, str(self.taps)),
                "seed (register state): " + str(self.register)]

    def __init__(self, length, taps, seed=0):
        """
        """

        self.length = length
        self.register = seed
        self.taps = taps

        super().__init__(seed)

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator.
        """
        super().seed(a, version)

        self.register = a

    def random_bit(self):
        # tap on the bits
        new_bit = lsb(reduce(xor, [self.register >> (p - 1) for p in self.taps]))

        last_bit = lsb(self.register)
        self.register = (new_bit << (self.length - 1)) | (self.register >> 1)
        # print('{1:0{0}b}'.format(self.length, self.register))

        return last_bit
