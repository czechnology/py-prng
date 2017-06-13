from generator.generator import NumberGenerator

"""
# References
[ONe15] M. E. O'Neill, "Pcg: A family of simple fast space-efficient statistically good algorithms
        for random number generation." Unpublished,
        http://www.pcg-random.org/pdf/toms-oneill-pcg-family-v1.02.pdf
[ONeWWW]M. E. O'Neill, "PCG, A Family of Better Random Number Generators" Online,
        http://www.pcg-random.org/
"""


class PermutedCongruentialGenerator(NumberGenerator):
    """An implementation of the PCG pseudorandom number generator, as proposed by O'Neill [ONe15].
    Ported from C implementation [OneWWW].
    """
    NAME = 'Permuted congruential generator'
    MASK32 = 0xffffffff
    MASK64 = 0xffffffffffffffff

    def state(self):
        return self.s

    def info(self):
        return [self.NAME,
                "parameter: seq_id=" + str(self.seq_id),
                "seed (state): " + str(self.state())]

    def __init__(self, init_seq, seed=None):
        self.s = None
        self.seq_id = None
        self.inc = None
        self._set_seq(init_seq)
        super().__init__(seed)

    def _set_seq(self, seq_id):
        self.seq_id = seq_id
        self.inc = (seq_id << 1) | 1

    def seed(self, a=None, version=2, seq=None):
        """Initialize the internal state of the generator."""

        if seq is not None:
            self._set_seq(seq)

        super().seed(a, version)

        self.s = 0
        self.random_number()
        self.s += a & self.MASK64
        self.random_number()

    def max_value(self):
        return 0xffffffff

    def random_number(self):

        if self.s is None:
            raise Exception("Generator must be initialized first")

        old_state = self.s
        self.s = (old_state * 6364136223846793005 + self.inc) & self.MASK64
        xor_shifted = (((old_state >> 18) ^ old_state) >> 27) & self.MASK32
        rot = old_state >> 59
        return ((xor_shifted >> rot) | (xor_shifted << ((-rot) & 31))) & self.MASK32

    def getstate(self):
        return self.VERSION, self.seq_id, self.s

    def setstate(self, state):
        self.VERSION = state[0]
        self._set_seq(state[1])
        self.s = state[2]
