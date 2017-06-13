from generator.generator import NumberGenerator

"""
# References
[Knu97] Donald E. Knuth. Art of Computer Programming, volume 2: Seminumerical Algorithms, chapter 3.
        Addison-Wesley Professional, 3rd edition, 1997.
"""


class LinearCongruentialGenerator(NumberGenerator):
    """A general implementation of a linear congruential generator as defined by Knuth [Knu97].
    Every number is generated by the formula

        X[n+1] = (a*X[n] + c) mod m,   n>=0,

    where m is the modulus (0<m), a is the multiplier (0<=a<m), c is the increment (0<=c<m),
    X[0] is the seed and X[n] is the sequence of generated numbers (n>0).
    """
    NAME = 'Linear congruential generator'

    def info(self):
        return [self.NAME,
                "parameters: m=%d, a=%d, c=%d" % (self.m, self.a, self.c),
                "seed (state): " + str(self.state())]

    def state(self):
        return self.x

    def __init__(self, m, a, c, seed=None):
        """Create the generator with specified constants.
        For practical purposes the constants should be a>=2 and b>=1  [Knu97].
        """
        self.m = m  # modulus
        self.a = a  # multiplier
        self.c = c  # increment
        self.x = 0
        super().__init__(seed)

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator.
        """
        super().seed(a, version)
        self.x = a

    def max_value(self):
        return self.m - 1

    def random_number(self):
        self.x = (self.a * self.x + self.c) % self.m
        return self.x

    def getstate(self):
        return self.VERSION, self.m, self.a, self.c, self.x

    def setstate(self, state):
        self.VERSION = state[0]
        self.m = state[1]
        self.a = state[2]
        self.c = state[3]
        self.x = state[4]


class JavaLinearCongruentialGenerator(LinearCongruentialGenerator):
    """
    LCG that is used in the standard Java's class java.util.Random.
    """
    M = 2 ** 48
    A = 25214903917
    C = 11

    def __init__(self, seed=None):
        super().__init__(self.M, self.A, self.C, seed)


class RanduLinearCongruentialGenerator(LinearCongruentialGenerator):
    """
    LCG that was used in the RANDU generator.
    """
    M = 2 ** 31
    A = 65539
    C = 0

    def __init__(self, seed=None):
        super().__init__(self.M, self.A, self.C, seed)
