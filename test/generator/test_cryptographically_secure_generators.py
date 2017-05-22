import operator
import unittest
from functools import reduce

from generator.cryptographically_secure_generators import BlumBlumShubGenerator
from utils.bit_tools import byte_to_bits
from utils.bit_tools import least_significant_bit as lsb


class TestCSPRNG(unittest.TestCase):
    def test_bbs_wiki(self):
        """Test if BBS PRBG correctly generates a simple sequence as described in Wikipedia
        example: https://en.wikipedia.org/wiki/Blum_Blum_Shub#Example """

        example_p = 11
        example_q = 19
        example_s = 3
        example_x = [9, 81, 82, 36, 42, 92]
        example_seq = [1, 1, 0, 0, 0, 0]

        self._test_bbs(gen_p=example_p, gen_q=example_q, seed=example_s,
                       expected_x=example_x, expected_bits=example_seq)

    def test_bbs_gawande(self):
        """Test if BBS PRBG correctly generates a simple sequence as described in paper by Gawande
        and Mundle (Kaustubh Gawande and Maithily Mundle. "Various Implementations of Blum Blum Shub
        Pseudo-Random Sequence Generator.
        http://cs.ucsb.edu/~koc/cren/project/pp/gawande-mundle.pdf """

        example_p = 7
        example_q = 19
        example_s = 2  # Gawande doesn't give the value of seed, only value of x0 = 4, so seed is 2
        example_x = [4, 16, 123, 100, 25, 93, 4, 16, 123, 100, 25, 93]  # period 6
        example_seq = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1]  # period 6

        self._test_bbs(gen_p=example_p, gen_q=example_q, seed=example_s,
                       expected_x=example_x, expected_bits=example_seq)

    def _test_bbs(self, gen_p, gen_q, seed, expected_x, expected_bits=None):
        if not expected_bits:
            expected_bits = [lsb(x) for x in expected_x]
        if len(expected_x) != len(expected_bits):
            raise ValueError("Lengths of expected x and bit values must be equal")

        rand = BlumBlumShubGenerator((gen_p, gen_q), seed=seed)
        self.assertEqual(rand.x, expected_x[0])

        bits = []
        for i in range(1, len(expected_x)):
            bit = rand.random_bit()
            bits.append(bit)
            self.assertEqual(bit, expected_bits[i])
            self.assertEqual(rand.x, expected_x[i])

        # test also the byte methods
        rand.seed(seed)
        bits_from_bytes = reduce(operator.add, [byte_to_bits(rand.random_byte()) for _ in
                                                range(len(expected_x) // 8 + 1)])
        self.assertListEqual(bits, bits_from_bytes[:len(bits)])


if __name__ == '__main__':
    unittest.main()
