import unittest
from time import perf_counter as time
from utils.unit_tools import nicer_time
from generator.true_randomness_generators import MicrophoneNoiseGenerator

from randomness_test import basic_tests as basic, fips_140_1_tests as fips


class TestStaticGenerators(unittest.TestCase):
    def test_microphone_noise_generator(self):
        generator = MicrophoneNoiseGenerator()
        # print("OUTPUT:", generator.random_bytes())

        n = 1000000//8

        # ts = time()
        # bytes_ = generator.random_bytes(n)
        # print(len(bytes_), "bytes in", nicer_time(time()-ts))

        ts = time()
        bytes_ = generator.random_bytes(n)
        print(len(bytes_), "bytes in", nicer_time(time()-ts))

        test_results = fips.run_all(generator, 10**6, print_log=True)
        print("FIPS tests result:", test_results)
        self.assertTrue(all(test_results))

        # test_results = basic.run_all(generator, 10**6, sig_level=0.01, print_log=True)
        # print("Basic tests result:", test_results)


if __name__ == '__main__':
    unittest.main()
