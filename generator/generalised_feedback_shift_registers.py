import abc

from generator.generator import NumberGenerator
from utils.bit_tools import least_significant_bit as lsb

"""
# References
[Mat92] Matsumoto, Makoto, and Yoshiharu Kurita. "Twisted GFSR generators." ACM Transactions on
        Modeling and Computer Simulation (TOMACS) 2.3 (1992): 179-194.
"""


class MersenneTwister(NumberGenerator, metaclass=abc.ABCMeta):
    # TODO: docs
    """ """

    def __init__(self, seed=None):
        """
        """

        # setup parameters according to the MT type
        self.w, self.n, self.m, self.r = (None, None, None, None)
        self.a = None
        self.u, self.d = None, None
        self.s, self.b = (None, None)
        self.t, self.c = (None, None)
        self.l = None
        self.f = None
        self.setup_parameters()

        self.lower_mask = (1 << self.r) - 1
        self.upper_mask = lsb(~self.lower_mask, self.w)

        # prepare fields
        self.mt = None
        self.index = None

        super().__init__(seed)

    @abc.abstractmethod
    def setup_parameters(self):
        raise NotImplementedError('Generators must define setup_parameters to use this base class')

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator.
        """
        super().seed(a, version)

        self.index = self.n
        self.mt = [0] * self.n
        self.mt[0] = a
        for i in range(1, self.n):
            self.mt[i] = lsb(self.f * (self.mt[i - 1] ^ (self.mt[i - 1] >> (self.w - 2))) + i,
                             self.w)

    def random_number(self):
        if self.mt is None or self.index is None:
            raise Exception("Generator must be initialized first")

        if self.index >= self.n:
            self._twist()

        y = self.mt[self.index]
        y = y ^ ((y >> self.u) & self.d)
        y = y ^ ((y << self.s) & self.b)
        y = y ^ ((y << self.t) & self.c)
        y = y ^ (y >> self.l)

        self.index += 1

        return lsb(y, self.w)

    def _twist(self):
        for i in range(self.n):
            x = (self.mt[i] & self.upper_mask) + (self.mt[(i + 1) % self.n] & self.lower_mask)
            x_a = x >> 1
            if x % 2 != 0:
                x_a = x_a ^ self.a
            self.mt[i] = self.mt[(i + self.m) % self.n] ^ x_a
        self.index = 0

    def max_value(self):
        return (2 ** self.w) - 1


class MersenneTwister32(MersenneTwister):
    NAME = 'Mersenne Twister (32-bit) generator'

    def info(self):
        return [self.NAME,
                "seed (state): MT[0]=" + str(self.mt[0])]

    def setup_parameters(self):
        self.w, self.n, self.m, self.r = (32, 624, 397, 31)
        self.a = 0x9908B0DF
        self.u, self.d = (11, 0xFFFFFFFF)
        self.s, self.b = (7, 0x9D2C5680)
        self.t, self.c = (15, 0xEFC60000)
        self.l = 18
        self.f = 1812433253


class MersenneTwister64(MersenneTwister):
    NAME = 'Mersenne Twister (64-bit) generator'

    def info(self):
        return [self.NAME,
                "seed (state): MT[0]=" + str(self.mt[0])]

    def setup_parameters(self):
        self.w, self.n, self.m, self.r = (64, 312, 156, 31)
        self.a = 0xB5026F5AA96619E9
        self.u, self.d = (29, 0x5555555555555555)
        self.s, self.b = (17, 0x71D67FFFEDA60000)
        self.t, self.c = (37, 0xFFF7EEE000000000)
        self.l = 43
        self.f = 6364136223846793005
