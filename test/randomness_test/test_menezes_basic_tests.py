import unittest
from os.path import dirname

from generator.generator import StaticSequenceGenerator, StaticFileGenerator
from randomness_test import basic_tests as basic
from randomness_test import fips_140_1_tests as fips


class TestMBT(unittest.TestCase):
    SEQ_FILE_PATH = dirname(__file__) + '/sequences/'
    SEQ_FILE_EXT = '.bin'
    SEQ_FILE_PATT = SEQ_FILE_PATH + '%s' + SEQ_FILE_EXT

    def test_menezes_example_statistic_values(self):
        """Test if the five basic tests behave as in the example given by Menezes"""

        sequence = [0b11100011,  # 227
                    0b00010001,  # 17
                    0b01001110,  # 78
                    0b11110010,  # 242
                    0b01001001]  # 73
        generator = StaticSequenceGenerator(seq=sequence)
        n = 160

        misc = {}

        generator.seed(0)
        x1 = basic.frequency_test(generator, n, misc=misc)
        self.assertEqual(misc['n0'], 84)
        self.assertEqual(misc['n1'], 76)
        self.assertEqual(x1, 0.4)

        generator.seed(0)
        x2 = basic.serial_test(generator, n, misc=misc)
        self.assertEqual(misc['n0'], 84)
        self.assertEqual(misc['n1'], 76)
        self.assertEqual(misc['n00'], 44)
        self.assertEqual(misc['n01'], 40)
        self.assertEqual(misc['n10'], 40)
        self.assertEqual(misc['n11'], 35)
        self.assertAlmostEqual(x2, 0.6252, places=4)

        generator.seed(0)
        x3 = basic.poker_test(generator, n, misc=misc)
        self.assertEqual(misc['m'], 3)
        self.assertEqual(misc['k'], 53)
        self.assertEqual(misc['ni'], [5, 10, 6, 4, 12, 3, 6, 7])
        self.assertAlmostEqual(x3, 9.6415, places=4)

        generator.seed(0)
        x4 = basic.runs_test(generator, n, misc=misc)
        self.assertEqual(misc['k'], 3)
        self.assertEqual(misc['e'], [20.25, 10.0625, 5])
        self.assertEqual(misc['b'], [25, 4, 5])
        self.assertEqual(misc['g'], [8, 20, 12])
        self.assertAlmostEqual(x4, 31.7913, places=4)

        generator.seed(0)
        x5 = basic.autocorrelation_test(generator, n, d=8, misc=misc)
        self.assertEqual(misc['a'], 100)
        self.assertAlmostEqual(x5, 3.8933, places=4)

        # print('Test (i):   {0:f}'.format(x1))
        # print('Test (ii):  {0:f}'.format(x2))
        # print('Test (iii): {0:f}'.format(x3))
        # print('Test (iv):  {0:f}'.format(x4))
        # print('Test (v):   {0:f}'.format(x5))

    def test_menezes_example_pass(self):
        """Test if the five basic tests behave as in the example given by Menezes"""

        sequence = [0b11100011,  # 227
                    0b00010001,  # 17
                    0b01001110,  # 78
                    0b11110010,  # 242
                    0b01001001]  # 73
        generator = StaticSequenceGenerator(seq=sequence)
        n = 160
        sig_level = 0.05

        generator.seed(0)
        pass1 = basic.frequency_test(generator, n, sig_level=sig_level)

        generator.seed(0)
        pass2 = basic.serial_test(generator, n, sig_level=sig_level)

        generator.seed(0)
        pass3 = basic.poker_test(generator, n, sig_level=sig_level)

        generator.seed(0)
        pass4 = basic.runs_test(generator, n, sig_level=sig_level)

        generator.seed(0)
        pass5 = basic.autocorrelation_test(generator, n, d=8, sig_level=sig_level)

        # print('Test (i):   {0:b}'.format(pass1))
        # print('Test (ii):  {0:b}'.format(pass2))
        # print('Test (iii): {0:b}'.format(pass3))
        # print('Test (iv):  {0:b}'.format(pass4))
        # print('Test (v):   {0:b}'.format(pass5))

        self.assertTrue(pass1)
        self.assertTrue(pass2)
        self.assertTrue(pass3)
        self.assertFalse(pass4)
        self.assertFalse(pass5)

    def test_menezes_fips186(self):
        """Test if the FIPS 186 against the example given by Menezes"""

        sequence = [0b11100011,  # 227
                    0b00010001,  # 17
                    0b01001110,  # 78
                    0b11110010,  # 242
                    0b01001001]  # 73
        generator = StaticSequenceGenerator(seq=sequence)
        n = 160

        t = fips.run_all(generator, n)
        # print(t1, t2, t3, t4)
        self.assertTupleEqual(t, (0, 0, 0, 1))

    def test_random(self):
        n = 16000
        sig_level = 0.05

        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'random_org_16384') as generator:
            generator.seed(0)
            misc = {}
            pass1 = basic.frequency_test(generator, n, sig_level=sig_level, misc=misc)

            generator.seed(0)
            misc = {}
            pass2 = basic.serial_test(generator, n, sig_level=sig_level, misc=misc)

            generator.seed(0)
            misc = {}
            pass3 = basic.poker_test(generator, n, sig_level=sig_level, misc=misc)

            generator.seed(0)
            misc = {}
            pass4 = basic.runs_test(generator, n, sig_level=sig_level, misc=misc)

            generator.seed(0)
            misc = {}
            pass5 = basic.autocorrelation_test(generator, n, d=8, sig_level=sig_level, misc=misc)

            self.assertTrue(all((pass1, pass2, pass3, pass4, pass5)))

            # FIPS 140-1
            generator.seed(0)
            t = fips.run_all(generator, n)
            self.assertTrue(all(t))

    def test_nist_examples(self):
        # 11001001000011111101101010100010001000010110100011
        # 00001000110100110001001100011001100010100010111000
        sequence = [0b11001001, 0b00001111, 0b11011010, 0b10100010, 0b00100001, 0b01101000,
                    0b11000010, 0b00110100, 0b11000100, 0b11000110, 0b01100010, 0b10001011,
                    0b10000000]  # last four bits get ignored

        generator = StaticSequenceGenerator(seq=sequence)
        n = 100

        generator.seed(0)
        misc = {}
        x1 = basic.frequency_test(generator, n, misc=misc)
        self.assertEqual(misc['n1'] - misc['n0'], -16)
        self.assertAlmostEqual(misc['p_value'], 0.109599, 6)

    def test_nist_sample_data_frequency(self):
        n = 10 ** 6
        sig_level = 0.05

        # Example #1: The binary expansion of pi
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/pi') as generator:
            misc = {}
            pass1 = basic.frequency_test(generator, n, sig_level=sig_level, misc=misc)
            self.assertAlmostEqual(misc['p_value'], 0.578211, 6)

        # Example #2: The binary expansion of e
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/e') as generator:
            misc = {}
            pass1 = basic.frequency_test(generator, n, sig_level=sig_level, misc=misc)
            self.assertAlmostEqual(misc['p_value'], 0.953749, 6)

        # Example #3: A G-SHA-1 binary sequence
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/sha1') as generator:
            misc = {}
            pass1 = basic.frequency_test(generator, n, sig_level=sig_level, misc=misc)
            self.assertAlmostEqual(misc['p_value'], 0.604458, 6)

        # Example #4: The binary expansion of sqrt(2)
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/sqrt2') as generator:
            misc = {}
            pass1 = basic.frequency_test(generator, n, sig_level=sig_level, misc=misc)
            self.assertAlmostEqual(misc['p_value'], 0.811881, 6)

        # Example #5: The binary expansion of sqrt(3)
        with StaticFileGenerator(file=self.SEQ_FILE_PATT % 'nist/sqrt3') as generator:
            misc = {}
            pass1 = basic.frequency_test(generator, n, sig_level=sig_level, misc=misc)
            self.assertAlmostEqual(misc['p_value'], 0.610051, 6)


if __name__ == '__main__':
    unittest.main()
