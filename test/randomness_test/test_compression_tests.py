import unittest
from os.path import dirname
from random import Random

from generator.generator import StaticSequenceGenerator, StaticFileGenerator
from generator.permuted_congruential_generators import PermutedCongruentialGenerator
from randomness_test.compression_tests import maurer_universal_test


class TestCompressionTests(unittest.TestCase):
    SEQ_FILE_PATH = dirname(__file__) + '/sequences/'
    SEQ_FILE_EXT = '.bin'
    SEQ_FILE_PATT = SEQ_FILE_PATH + '%s' + SEQ_FILE_EXT

    def test_maurer_test_random(self):
        """Test if the random sample from random.org passes the test"""
        n = 128000
        sig_level = 0.001

        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'random_org_16384') as generator:
            misc = {}
            t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
            print("Maurer's universal statistical test:", t, misc)
            self.assertTrue(t)

    def test_maurer_test_prng(self):
        n = 10 ** 6
        sig_level = 0.01

        # generator = StaticFileGenerator(file='sequences/mt32-100M.bin')
        # misc = {}
        # t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
        # print("Maurer's universal statistical test for MT32:", t, misc)
        # self.assertTrue(t)
        #
        # generator = StaticFileGenerator(file='sequences/mt64-100M.bin')
        # misc = {}
        # t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
        # print("Maurer's universal statistical test for MT64:", t, misc)
        # self.assertTrue(t)

        # generator = StaticFileGenerator(file='sequences/pcg-100M.bin')
        # misc = {}
        # t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
        # print("Maurer's universal statistical test for PCG:", t, misc)
        # # self.assertTrue(t)

        r = Random(123)
        generator = PermutedCongruentialGenerator(r.getrandbits(159), seed=r.getrandbits(160))
        misc = {}
        t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
        print("Maurer's universal statistical test for PCG:", t, misc)
        self.assertTrue(t)

        # generator = StaticFileGenerator(file='sequences/lcg_java-100M.bin')
        # misc = {}
        # t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
        # print("Maurer's universal statistical test for Java's LCG:", t, misc)
        # self.assertTrue(t)
        #
        # generator = create_generator('lcg_simple')
        # misc = {}
        # t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
        # print("Maurer's universal statistical test for a very simple LCG:", t, misc)
        # self.assertFalse(t)
        #
        # generator = create_generator('lcg_randu')
        # misc = {}
        # t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
        # print("Maurer's universal statistical test for RANDU LCG:", t, misc)
        # self.assertFalse(t)

        # generator = StaticFileGenerator(file='sequences/zero.bin')
        # misc = {}
        # t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
        # print("Maurer's universal statistical test for zeros:", t, misc)
        # self.assertFalse(t)

        # generator = StaticFileGenerator(file='sequences/almost-zero.bin')
        # misc = {}
        # t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
        # print("Maurer's universal statistical test for almost zeros:", t, misc)
        # self.assertFalse(t)

    def test_nist_examples(self):
        # 01011010 01110101 0111
        sequence = [0b01011010, 0b01110101, 0b01110000]  # last four bits get ignored

        n = 20
        generator = StaticSequenceGenerator(seq=sequence)

        generator.seed(0)
        misc = {}
        maurer_universal_test(generator, n, l=2, q=4, misc=misc)
        print("Universal:", misc)
        self.assertEqual(misc['k'], 6)
        self.assertAlmostEqual(misc['xu'], 1.1949875, 8)
        # self.assertAlmostEqual(misc['p_value'], 0.767189, 6)  # TODO: verify

        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/sha1') as generator:
            misc = {}
            maurer_universal_test(generator, 1048576, l=7, q=1280, misc=misc)
            # Expected values:  c=0.591311, sigma=0.002703, K=148516, sum=919924.038020
            #                   xu=fn=6.194107, expectedValue=6.196251, sigma=3.125
            #                   P-value = 0.427733
            print("Universal:", misc)
            # TODO: assert (after implementing the generator)

    def test_maurer_nist_sample_data_1(self):
        n = 10 ** 6
        sig_level = 0.05

        # Example #1: The binary expansion of pi
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/pi') as generator:
            misc = {}
            t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
            self.assertAlmostEqual(misc['p_value'], 0.669012, 3)

    def test_maurer_nist_sample_data_2(self):
        n = 10 ** 6
        sig_level = 0.05

        # Example #2: The binary expansion of e
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/e') as generator:
            misc = {}
            t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
            self.assertAlmostEqual(misc['p_value'], 0.282568, 3)

    def test_maurer_nist_sample_data_3(self):
        n = 10 ** 6
        sig_level = 0.05

        # Example #3: A G-SHA-1 binary sequence
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/sha1') as generator:
            misc = {}
            t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
            self.assertAlmostEqual(misc['p_value'], 0.411079, 3)

    def test_maurer_nist_sample_data_4(self):
        n = 10 ** 6
        sig_level = 0.05

        # Example #4: The binary expansion of sqrt(2)
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/sqrt2') as generator:
            misc = {}
            t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
            print("Universal test of sqrt(2):", t, misc)
            self.assertAlmostEqual(misc['p_value'], 0.130805, 3)

    def test_maurer_nist_sample_data_5(self):
        n = 10 ** 6
        sig_level = 0.05

        # Example #5: The binary expansion of sqrt(3)
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/sqrt3') as generator:
            misc = {}
            t = maurer_universal_test(generator, n, sig_level=sig_level, misc=misc)
            print("Universal test of sqrt(3):", t, misc)
            self.assertAlmostEqual(misc['p_value'], 0.165981, 3)


if __name__ == '__main__':
    unittest.main()
