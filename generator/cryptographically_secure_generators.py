from math import gcd
from random import SystemRandom as Random

from generator.generator import BitGenerator
from utils.bit_tools import least_significant_bit as lsb
from utils.prime_tools import is_prime

"""
# References
[Men96] Alfred J Menezes, Paul C Van Oorschot, and Vanstone. Handbook of applied cryptography,
        chapter 5. CRC press, 1996.
"""


class RsaGenerator(BitGenerator):
    """An implementation of the RSA pseudorandom bit generator, as described by Menezes [Men96].
    """
    NAME = 'RSA generator'

    def info(self):
        return [self.NAME,
                "parameters: p=%d, q=%d, n=%d, e=%d" % (self.p, self.q, self.n, self.e),
                "seed (state): " + str(self.x)]

    def __init__(self, pqe=None, seed=None):
        if pqe is not None:
            if len(pqe) != 3:
                raise ValueError("Parameter pqn must be a triple of the values p, q and e")
            self.p, self.q, self.e = pqe
            self.n = self.p * self.q
            self._verify_params()
        else:
            self._gen_params(511)

        self.x = None

        super().__init__(seed)

    def _gen_params(self, bits):
        # use builtin RNG
        rand = Random()

        self.p = rand.getrandbits(bits)
        while not is_prime(self.p):
            self.p = rand.getrandbits(bits)

        self.q = rand.getrandbits(bits)
        while not is_prime(self.q):
            self.q = rand.getrandbits(bits)

        self.n = self.p * self.q

        self._gen_param_e()

        try:
            self._verify_params()
        except ValueError as err:
            raise Exception("Some of the p, q, e values were generated incorrectly: " + str(err))

    def _gen_param_e(self):
        # use builtin RNG
        rand = Random()

        phi = (self.p - 1) * (self.q - 1)
        self.e = rand.randint(2, phi - 1)
        while gcd(self.e, phi) != 1:
            self.e = rand.randint(2, phi - 1)

    def _verify_params(self):
        phi = (self.p - 1) * (self.q - 1)

        if self.p == self.q:
            raise ValueError("p and q cannot be identical")
        # verify that p and q are far enough apart - https://crypto.stackexchange.com/a/35096
        if self.n >= 1024 and abs(self.p - self.q).bit_length() <= (self.n.bit_length() // 2 - 100):
            raise ValueError("p and q are too close together")
        if not is_prime(self.p):
            raise ValueError("p must be a prime (probabilistic)")
        if not is_prime(self.q):
            raise ValueError("q must be a prime (probabilistic)")
        if not (1 < self.e < phi):
            raise ValueError("e must be between 1 and (p-1)(q-1)")
        if gcd(self.e, phi) != 1:
            raise ValueError("e must be co-prime to (p-1)(q-1)")

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator."""
        if not (1 <= a <= self.n - 1):
            raise ValueError("Seed value must be between 1 and n-1=" + str(self.n - 1))
        super().seed(a, version)
        self.x = a

    def random_bit(self):
        # Generate pseudorandom bit as per [Men96] (sequence with length l=1)
        # SUMMARY: a pseudorandom bit sequence z1,z2,...,zl of length l is generated.
        # 3.    For i from 1 to l do the following:
        # 3.1   x[i] = x[i-1]^e mod n.
        # 3.2   z[i] = the least significant bit of x[i].
        # 4.    The output sequence is z1,z2,...,zl.

        self.x = pow(self.x, self.e, self.n)
        z = self.x & 1  # least significant bit of x

        return z

    @staticmethod
    def xor(ba1, ba2):
        return [b1 ^ b2 for (b1, b2) in zip(ba1, ba2)]

    def getstate(self):
        return self.VERSION, self.p, self.q, self.e, self.x

    def setstate(self, state):
        self.VERSION = state[0]
        self.p = state[1]
        self.q = state[2]
        self.n = self.p * self.q
        self.e = state[3]
        self.x = state[4]


class MicaliSchnorrGenerator(RsaGenerator, BitGenerator):
    NAME = 'Micali-Schnorr generator'

    def info(self):
        return [self.NAME,
                "parameters: p=%d, q=%d, n=%d, e=%d, k=%d, r=%d" % (self.p, self.q, self.n, self.e,
                                                                    self.k, self.r),
                "seed (state): " + str(self.x)]

    def __init__(self, pqe=None, seed=None):
        self.r = 0
        super().__init__(pqe, seed)
        n_bits = self.n.bit_length()
        self.k = int(n_bits * (1 - 2 / self.e))
        self.r = n_bits - self.k

    def _gen_param_e(self):
        # use builtin RNG
        rand = Random()
        n_bits = self.n.bit_length()

        phi = (self.p - 1) * (self.q - 1)
        mx = min(phi - 1, n_bits // 80)  # top limits are phi (exclusive) and N/80 (inclusive)
        self.e = rand.randint(2, mx)
        while gcd(self.e, phi) != 1:
            self.e = rand.randint(2, mx)

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator."""
        if self.r and a.bit_length() != self.r:
            raise ValueError("Seed value must be r=" + str(self.r) + " bits long")
        super().seed(a, version)
        self.x = a

    def generate_to_buffer(self):
        # Generate pseudorandom bit sequence as per [Men96]
        # SUMMARY: a pseudorandom bit sequence is generated.
        # 3.    Generate a pseudorandom sequence of length k*l.
        #       For i from 1 to l do the following:
        # 3.1       y[i] = x[i-1]^e mod n.
        # 3.2       x[i] = the r most significant bits of y[i].
        # 3.3       z[i] = the k least significant bits of y[i].
        # 4.    The output sequence is z1||z2||...||zl, where || denotes concatenation

        y = pow(self.x, self.e, self.n)
        self.x = lsb(y >> self.k, self.r)  # r=N-k most significant bits of y
        z = lsb(y, self.k)  # k least significant bits of y

        # # collect k bits from z to a list - based on http://stackoverflow.com/a/10322122
        # bits_list = [(z >> i) & 1 for i in range(self.k - 1, -1, -1)]

        self.add_to_buffer(z, self.k)

    def getstate(self):
        return self.VERSION, self.p, self.q, self.e, self.r, self.k, self.x

    def setstate(self, state):
        self.VERSION = state[0]
        self.p = state[1]
        self.q = state[2]
        self.n = self.p * self.q
        self.e = state[3]
        self.r = state[4]
        self.k = state[5]
        self.x = state[6]


class BlumBlumShubGenerator(BitGenerator):
    """An implementation of the Blum Blum Shub pseudorandom bit generator, as described by Menezes
    [Men96].
    """
    NAME = 'Blum-Blum-Shub generator'

    def info(self):
        return [self.NAME,
                "parameters: p=%d, q=%d, n=%d" % (self.p, self.q, self.n),
                "seed (state): " + str(self.x)]

    def __init__(self, pq=None, seed=None):
        if pq is not None:
            if len(pq) != 2:
                raise ValueError("Parameter pq must be a triple of the values p and q")
            self.p, self.q = pq
            self.n = self.p * self.q
            self._verify_params()
        else:
            self._gen_params(511)

        self.x = None

        if not seed:
            seed = Random().randrange(1, self.n)
            while gcd(seed, self.n) != 1:
                seed = Random().randrange(1, self.n)

        super().__init__(seed)

    def _gen_params(self, bits):
        # use builtin RNG
        rand = Random()

        self.p = rand.getrandbits(bits)
        while not (is_prime(self.p) and self.p % 4 == 3):
            self.p = rand.getrandbits(bits)

        self.q = rand.getrandbits(bits)
        while not (is_prime(self.q) and self.q % 4 == 3):
            self.q = rand.getrandbits(bits)

        self.n = self.p * self.q

        try:
            self._verify_params()
        except ValueError as err:
            raise Exception("Some of the p or q values were generated incorrectly: " + str(err))

    def _verify_params(self):

        if self.p == self.q:
            raise ValueError("p and q cannot be identical")
        # verify that p and q are far enough apart - https://crypto.stackexchange.com/a/35096
        if self.n >= 1024 and abs(self.p - self.q).bit_length() <= (self.n.bit_length() // 2 - 100):
            raise ValueError("p and q are too close together")
        if not is_prime(self.p):
            raise ValueError("p must be a prime (probabilistic)")
        if not is_prime(self.q):
            raise ValueError("q must be a prime (probabilistic)")
        if self.p % 4 != 3:
            raise ValueError("p must be congruent to 3 modulo 4")
        if self.q % 4 != 3:
            raise ValueError("q must be congruent to 3 modulo 4")

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator."""
        if not (1 <= a <= self.n - 1):
            raise ValueError("Seed value must be between 1 and n-1=" + str(self.n - 1))
        if gcd(a, self.n) != 1:
            raise ValueError("Seed value must be co-prime to n=" + str(self.n))

        super().seed(a, version)
        self.x = pow(a, 2, self.n)

    def random_bit(self):
        # Generate pseudorandom bit as per [Men96] (sequence with length l=1)

        # SUMMARY: a pseudorandom bit sequence z1,z2,...,zl of length l is generated.
        # 3.    For i from 1 to l do the following:
        # 3.1       x[i] = x[i-1]^2 mod n.
        # 3.2       z[i] = lsb(x[i]).
        # 4.    The output sequence is z1,z2,...,zl

        self.x = pow(self.x, 2, self.n)
        z = self.x & 1  # least significant bit of x

        return z
