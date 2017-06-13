import unittest

from generator.one_way_function_generators import \
    AnsiX917Generator, Fips186Generator, Fips186GeneratorPk, Fips186GeneratorMsg
from randomness_test import fips_140_1_tests as fips


class TestOneWayFunctionGenerators(unittest.TestCase):
    SEQUENCES_PATH = "sequences/one_way_function_generators"

    def test_ansix917(self):
        """Currently no test vectors available, test general function for now"""

        generator = AnsiX917Generator(seed=123456789)
        print(fips.run_all(generator, print_log=True))

    def test_fips186_pk(self):
        """Currently no test vectors available, test general function for now"""

        generator = Fips186GeneratorPk(g=Fips186Generator.OWF_SHA1, seed=123456789)
        print(fips.run_all(generator, print_log=True))

        generator = Fips186GeneratorPk(g=Fips186Generator.OWF_DES, seed=123456789)
        print(fips.run_all(generator, print_log=True))

    def test_fips186_msg(self):
        """Currently no test vectors available, test general function for now"""

        generator = Fips186GeneratorMsg(g=Fips186Generator.OWF_SHA1, seed=123456789)
        print(fips.run_all(generator, print_log=True))

        generator = Fips186GeneratorMsg(g=Fips186Generator.OWF_DES, seed=123456789)
        print(fips.run_all(generator, print_log=True))


if __name__ == '__main__':
    unittest.main()
