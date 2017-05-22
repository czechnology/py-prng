import unittest

from generator import linear_congruential_generators as lcg
from utils.bit_tools import least_significant_bit as lsb


class TestLCG(unittest.TestCase):
    SEQUENCES_PATH = "sequences/linear_congruential_generators"

    def test_knuth_lcg_3_2_1(self):
        """Test if LCG correctly generates a simple sequence as described by Knuth (p.10)"""

        self._test_lcg(
            gen_m=10,
            gen_a=7,
            gen_c=7,
            seed=7,
            expected_x=[6, 9, 0, 7, 6, 9, 0, 7]  # period 4
        )

    def test_java_random(self):
        """Java's java.util.Random uses a LCG with parameters m = 2**48, a = 25214903917, c = 11,
        while only using the bits 47..16
        Test against pre-generated values"""
        rand = lcg.JavaLinearCongruentialGenerator()

        rand.seed(0)
        generated_sequence = [lsb(rand.random_number() >> 16, 32) for _ in range(1000)]
        expected_sequence = self._read_sequence(self.SEQUENCES_PATH + "/java-s0.txt")
        self.assertEqual(generated_sequence, expected_sequence)

        rand.seed(123)
        generated_sequence = [lsb(rand.random_number() >> 16, 32) for _ in range(1000)]
        expected_sequence = self._read_sequence(self.SEQUENCES_PATH + "/java-s123.txt")
        self.assertEqual(generated_sequence, expected_sequence)

        rand.seed(1088542510)
        generated_sequence = [lsb(rand.random_number() >> 16, 32) for _ in range(1000)]
        expected_sequence = self._read_sequence(self.SEQUENCES_PATH + "/java-s1088542510.txt")
        self.assertEqual(generated_sequence, expected_sequence)

    def test_randu(self):
        """Test if LCG generates the sequence as a RANDU generator"""

        rand = lcg.RanduLinearCongruentialGenerator(1)
        generated_sequence = [rand.random_number() for _ in range(20)]
        expected_sequence = self._read_sequence(self.SEQUENCES_PATH + "/randu-s1.txt")
        self.assertListEqual(generated_sequence, expected_sequence)

    def _test_lcg(self, gen_m, gen_a, gen_c, seed, expected_x):
        if type(expected_x) is str:
            expected_x = self._read_sequence(self.SEQUENCES_PATH + '/' + '.txt')

        rand = lcg.LinearCongruentialGenerator(m=gen_m, a=gen_a, c=gen_c)
        rand.seed(seed)

        for i in range(len(expected_x)):
            num = rand.random_number()
            self.assertEqual(num, expected_x[i])

    @staticmethod
    def _read_sequence(file):
        with open(file) as f:
            sequence = f.readlines()

        sequence = list(filter(lambda l: l[:1] != "#", sequence))
        sequence = list(map(lambda l: int(l.strip()), sequence))

        return sequence


if __name__ == '__main__':
    unittest.main()
