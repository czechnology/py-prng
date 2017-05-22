import abc
import time
from random import Random

from pyDes import des, triple_des

from generator.generator import NumberGenerator
from utils.bit_tools import byte_xor, split_chunks, concat_chunks, left_rotate, \
    least_significant_bit as lsb
from utils.prime_tools import is_prime

"""
# References
[Men96] Alfred J Menezes, Paul C Van Oorschot, and Vanstone. Handbook of applied cryptography,
        chapter 5. CRC press, 1996.
"""


class AnsiX917Generator(NumberGenerator):
    """An implementation of the ANSI X9.17 generator, as described by Menezes et al [Men96]:

    5.11 Algorithm ANSI X9.17 pseudorandom bit generator
    INPUT: a random (and secret) 64-bit seed s, integer m, and DES E-D-E encryption key k.
    OUTPUT: m pseudorandom 64-bit strings x[1],x[2],...,x[m].
    1.  Compute the intermediate value I = Ek(D), where D is a 64-bit representation of
        the date/time to as fine a resolution as is available.
    2.  For i from 1 to m do the following:
    2.1     x[i] = Ek(I xor s).
    2.2     s    = Ek(x[i] xor I).
    3.  Return(x[1],x[2],...,x[m]).
    """

    NAME = "ANSI X9.17 Generator"
    DEFAULT_KEY = b"qjIfOAWOoYFnaNjg"

    def info(self):
        return [self.NAME,
                "parameters: key=%s, d=%s" % (self.key, str(self.d)),
                "seed (state): " + str(self.s)]

    def __init__(self, seed=None, key=None, t=None):
        self.s = seed  # state
        self.key = key if key else self.DEFAULT_KEY
        self.tdea = triple_des(self.key).encrypt  # encryption algorithm

        # Compute the intermediate value I = Ek(D), where D is a 64-bit representation of the
        # date/time to as fine a resolution as is available.
        if t is None:
            t = int(time.time() * (2 ** 31))
        d = t.to_bytes(8, byteorder='big')
        self.i = self.tdea(d)

        super().__init__(seed)

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator.
        """
        super().seed(a, version)

        if a is None:
            return

        self.s = a.to_bytes(8, byteorder='big')

    def max_value(self):
        return (1 << 64) - 1

    def random_number(self, m=None):
        """Generate pseudorandom number as per [Men96]:
        
        2.  For i from 1 to m do the following:
        2.1     x[i] = Ek(I xor s).
        2.2     s    = Ek(x[i] xor I).
        3.  Return(x[1],x[2],...,x[m]).
        """

        x = self.tdea(byte_xor(self.i, self.s))
        self.s = self.tdea(byte_xor(x, self.i))

        return int.from_bytes(x, byteorder='big')

    def getstate(self):
        return self.VERSION, self.s

    def setstate(self, state):
        self.VERSION = state[0]
        self.s = state[1]


class Fips186Generator(NumberGenerator, metaclass=abc.ABCMeta):
    OWF_SHA1 = 1
    OWF_DES = 2

    def __init__(self, g=OWF_SHA1, q=None, seed=None, b=160):
        self.s = seed  # state
        self.b = b  # exponent
        self.q = q

        if q is None:
            self._gen_param_q()

        if g == self.OWF_SHA1:
            self.g = self._one_way_sha1
        elif g == self.OWF_DES:
            self.g = self._one_way_des
        else:
            raise ValueError("Wrong one-way function parameter")

        self._verify_params(g)

        super().__init__(None)

    def _gen_param_q(self):
        # use builtin RNG
        rand = Random()

        self.q = rand.getrandbits(160)
        while not is_prime(self.q):
            self.q = rand.getrandbits(160)

    def _verify_params(self, g):
        if not is_prime(self.q):
            raise ValueError("Parameter q must be a prime (probabilistic)")
        if self.q.bit_length() > 160:
            raise ValueError("Parameter q must be a 160-bit integer")
        if g == self.OWF_SHA1 and not (160 <= self.b <= 512):
            raise ValueError("Parameter b must be between 160 and 512 for use with SHA-1")
        elif g == self.OWF_DES and self.b != 160:
            raise ValueError("Parameter b must be 160 for use with DES")

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator.
        The seed value should be a random b-bit integer.
        """
        super().seed(a, version)
        self.s = a

    def max_value(self):
        return self.q - 1

    @classmethod
    def _one_way_sha1(cls, t, c, b):

        """Special one-way function based on SHA-1 for use in FIPS 186 PRNG as per [Men96]:
        
        5.15 Algorithm FIPS 186 one-way function using SHA-1
        INPUT: a 160-bit string t and a b-bit string c, 160 <= b <= 512.
        OUTPUT: a 160-bit string denoted G(t,c).
        1. Break up t into five 32-bit blocks: t = H1||H2||H3||H4||H5.
        2. Pad c with 0's to obtain a 512-bit message block: X = c || 0^{512-b}.
        3. Divide X into 16 32-bit words: x[0]x[1]...x[15], and set m=1.
        4. Execute step 4 of SHA-1 (Algorithm 9.53). (This alters the H[i]'s.)
        5. The output is the concatenation: G(t,c) = H1||H2||H3||H4||H5.
        """

        # print(bin(t))

        # 1. Break up t into five 32-bit blocks: t = H1||H2||H3||H4||H5.
        h = split_chunks(t, 32, pad=5)

        # 2. Pad c with 0's to obtain a 512-bit message block: X = c || 0^{512-b}.
        # 3. Divide X into 16 32-bit words: x[0]x[1]...x[15], and set m=1.
        x = split_chunks(c << (512 - b), 32, pad=16)

        # 4. Execute step 4 of SHA-1 (Algorithm 9.53). (This alters the H[i]'s.)
        h = cls._sha1_compression_function(x, h)

        # 5. The output is the concatenation: G(t,c) = H1||H2||H3||H4||H5.
        return concat_chunks(h, 32)

    @classmethod
    def _sha1_compression_function(cls, x, h):
        assert (len(x) == 16)
        assert (len(h) == 5)

        # Define per-round integer additive constants:
        y1, y2, y3, y4 = 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6

        # function f as in MD4:  f(u,v,w) = uv or (~u)w
        def ff(u, v, w):
            return (u & v) | ((~u) & w)

        # function g as in MD4:  g(u,v,w) = uv or uw or vw
        def fg(u, v, w):
            return (u & v) | (u & w) | (v & w)

        # function h as in MD4:  h(u,v,w) = u xor v xor w
        def fh(u, v, w):
            return u ^ v ^ w

        # (expand 16-word block into 80-word block)
        # for j from 16 to 79, X[j] = ((X[j-3] xor X[j-8] xor X[j-14] xor X[j-16]) left_rotate 1).
        x += [0] * 64  # expand list
        for j in range(16, 80):
            x[j] = left_rotate(x[j - 3] ^ x[j - 8] ^ x[j - 14] ^ x[j - 16], 1, bits=32)

        # (initialize working variables)
        # (A, B, C, D, E) = (H[1], H[2], H[3], H[4], H[5]).
        a, b, c, d, e = h

        # (Round 1) For j from 0 to 19 do the following:
        # t = (A left_rot 5) + f(B, C, D) + E + X[j] + y1,
        # (A, B, C, D, E) = (t, A, B left_rot 30, C, D).
        # (Round 2) For j from 20 to 39 do the following:
        # t = (A left_rot 5) + h(B, C, D) + E + X[j] + y2,
        # (A, B, C, D, E) = (t, A, B left_rot 30, C, D).
        # (Round 3) For j from 40 to 59 do the following:
        # t = (A left_rot 5) + g(B, C, D) + E + X[j] + y3,
        # (A, B, C, D, E) = (t, A, B left_rot 30, C, D).
        # (Round 4) For j from 60 to 79 do the following:
        # t = (A left_rot 5) + h(B, C, D) + E + X[j] + y4,
        # (A, B, C, D, E) = (t, A, B left_rot 30, C, D).
        for j in range(0, 80):
            if j < 20:  # Round 1
                func_res = ff(b, c, d)
                y = y1
            elif j < 40:  # Round 2
                func_res = fh(b, c, d)
                y = y2
            elif j < 60:  # Round 3
                func_res = fg(b, c, d)
                y = y3
            else:  # Round 4
                func_res = fh(b, c, d)
                y = y4

            t = lsb(left_rotate(a, 5, bits=32) + func_res + e + x[j] + y, 32)
            (a, b, c, d, e) = (t, a, left_rotate(b, 30, bits=32), c, d)

        # (update chaining values)
        # (H1, H2, H3, H4, H5) = (H1+A, H2+B, H3+C, H4+D, H5+E).
        h = (
            lsb(h[0] + a, 32),
            lsb(h[1] + b, 32),
            lsb(h[2] + c, 32),
            lsb(h[3] + d, 32),
            lsb(h[4] + e, 32)
        )
        return h

    @classmethod
    def _one_way_des(cls, t, c):
        """Special one-way function based on DES for use in FIPS 186 PRNG as per [Men96]:

        5.16 Algorithm FIPS 186 one-way function using DES
        INPUT: two 160-bit strings t and c.
        OUTPUT: a 160-bit string denoted G(t;c).
        1.  Break up t into five 32-bit blocks: t = t0||t1||t2||t3||t4.
        2.  Break up c into five 32-bit blocks: c = c0||c1||c2||c3||c4.
        3.  For i from 0 to 4 do the following: x[i] = t[i] xor c[i].
        4.  For i from 0 to 4 do the following:
        4.1     b1 = c[i+4 mod 5], b2 = c[i+3 mod 5].
        4.2     a1 = x[i], a2 = x[i+1 mod 5] xor x[i+4 mod 5].
        4.3     A = a1||a2, B = b'[1]||b2,
                    where b'[1] denotes the 24 least significant bits of b1.
        4.4     Use DES with key B to encrypt A: y[i] = DES_B(A).
        4.5     Break up y[i] into two 32-bit blocks: yi = L[i]||R[i].
        5. For i from 0 to 4 do the following: z[i] = L[i] xor R[i+2 mod 5] xor L[i+3 mod 5].
        6. The output is the concatenation: G(t,c) = z0||z1||z2||z3||z4.
        """
        b = 160

        # 1.  Break up t into five 32-bit blocks: t = t0||t1||t2||t3||t4.
        t = split_chunks(t, 32, 5)

        # 2.  Break up c into five 32-bit blocks: c = c0||c1||c2||c3||c4.
        c = split_chunks(c, 32, 5)

        # 3.  For i from 0 to 4 do the following: x[i] = t[i] xor c[i].
        x = list(map(lambda pair: pair[0] ^ pair[1], zip(t, c)))

        # 4.  For i from 0 to 4 do the following:
        l = r = [0] * 5
        for i in range(5):
            # 4.1     b1 = c[i+4 mod 5], b2 = c[i+3 mod 5].
            # 4.2     a1 = x[i], a2 = x[i+1 mod 5] xor x[i+4 mod 5].
            b1 = c[(i + 4) % 5]
            b2 = c[(i + 3) % 5]
            a1 = x[i]
            a2 = x[(i + 1) % 5] ^ x[(i + 4) % 5]

            # 4.3     A = a1||a2, B = b'[1]||b2,
            #             where b'[1] denotes the 24 least significant bits of b1.
            a = concat_chunks((a1, a2), 32)
            b = concat_chunks((lsb(b1, 24), b2), 32)

            # 4.4     Use DES with key B to encrypt A: y[i] = DES_B(A).
            y = cls._des_encrypt(key=b, msg=a)

            # 4.5     Break up y[i] into two 32-bit blocks: yi = L[i]||R[i].
            l[i], r[i] = split_chunks(y, bits=32, pad=2)

        # 5. For i from 0 to 4 do the following: z[i] = L[i] xor R[i+2 mod 5] xor L[i+3 mod 5].
        z = [l[i] ^ r[(i + 2) % 5] ^ l[(i + 3) % 5] for i in range(5)]

        # 6. The output is the concatenation: G(t,c) = z0||z1||z2||z3||z4.
        return concat_chunks(z, bits=32)

    @classmethod
    def _des_encrypt(cls, key, msg):
        key_bytes = key.to_bytes(8, byteorder='big')
        msg_bytes = msg.to_bytes(8, byteorder='big')
        encrypted = des(key=key_bytes).encrypt(data=msg_bytes)
        return int.from_bytes(encrypted, byteorder='big')


class Fips186GeneratorPk(Fips186Generator):
    NAME = "FIPS 186 Generator For DSA Private Keys"

    def info(self):
        g = 'SHA1' if (self.g == self._one_way_sha1) else 'DES'
        return [self.NAME,
                "parameters: b=%d, q=%d, g=%s" % (self.b, self.q, g),
                "seed (state): " + str(self.s)]

    def random_number(self):
        # 3.    Define the 160-bit string t = 67452301 efcdab89 98badcfe 10325476 c3d2e1f0 (hex).
        # 4.    For i from 1 to m do the following:
        #   4.1   (optional user input) Either select a b-bit string y[i], or set y[i]=0.
        #   4.2   z[i] = (s + y[i]) mod 2^b.
        #   4.3   a[i] = G(t, z[i]) mod q.
        #   4.4   s = (1 + s + a[i]) mod 2^b
        # 5.    Return (a1,a2,...,am).

        t = 0x67452301efcdab8998badcfe10325476c3d2e1f0

        # TODO: (optional user input) Either select a b-bit string y[i], or set y[i]=0.
        y = 0
        z = lsb(self.s + y, self.b)
        a = self.g(t, z, self.b) % self.q
        self.s = lsb(1 + self.s + a, self.b)

        return a


class Fips186GeneratorMsg(Fips186Generator):
    NAME = "FIPS 186 Generator For DSA Per-Message Secrets"

    def info(self):
        g = 'SHA1' if (self.g == self._one_way_sha1) else 'DES'
        return [self.NAME,
                "parameters: b=%d, q=%d, g=%s" % (self.b, self.q, g),
                "seed (state): " + str(self.s)]

    def random_number(self):
        # 3.    Define the 160-bit string t = efcdab89 98badcfe 10325476 c3d2e1f0 67452301 (hex).
        # 4.    For i from 1 to m do the following:
        #   4.1     k[i] = G(t,s) mod q.
        #   4.2     s = (1 + s + k[i]) mod 2^b.
        # 5.    Return (k[1],k[2],...,k[m]).

        t = 0xefcdab8998badcfe10325476c3d2e1f067452301

        k = self.g(t, self.s, self.b) % self.q
        self.s = lsb(1 + self.s + k, self.b)

        return k
