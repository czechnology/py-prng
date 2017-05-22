import unittest

from generator.generator import StaticSequenceGenerator, StaticFileGenerator


class TestStaticGenerators(unittest.TestCase):
    def test_static_sequence_generator(self):
        """Test the static sequence generator"""

        sequence = [0b11100011,  # 227
                    0b00010001,  # 17
                    0b01001110,  # 78
                    0b11110010,  # 242
                    0b01001001]  # 73

        generator = StaticSequenceGenerator(seq=sequence)

        generated_sequence = [generator.random_byte() for _ in range(10)]
        self.assertEqual(generated_sequence, sequence * 2)

        generated_sequence = [generator.random_bit() for _ in range(80)]
        self.assertEqual(generated_sequence, [1, 1, 1, 0, 0, 0, 1, 1,
                                              0, 0, 0, 1, 0, 0, 0, 1,
                                              0, 1, 0, 0, 1, 1, 1, 0,
                                              1, 1, 1, 1, 0, 0, 1, 0,
                                              0, 1, 0, 0, 1, 0, 0, 1] * 2)

    def test_static_file_generator(self):
        """Test the static file generator"""

        sequence = [0b11100011,  # 227
                    0b00010001,  # 17
                    0b01001110,  # 78
                    0b11110010,  # 242
                    0b01001001]  # 73

        with StaticFileGenerator(
                file='../randomness_test/sequences/menezes_example.bin') as generator:
            generated_sequence = [generator.random_byte() for _ in range(10)]
            self.assertEqual(generated_sequence, sequence * 2)

            generated_sequence = [generator.random_bit() for _ in range(80)]
            self.assertEqual(generated_sequence, [1, 1, 1, 0, 0, 0, 1, 1,
                                                  0, 0, 0, 1, 0, 0, 0, 1,
                                                  0, 1, 0, 0, 1, 1, 1, 0,
                                                  1, 1, 1, 1, 0, 0, 1, 0,
                                                  0, 1, 0, 0, 1, 0, 0, 1] * 2)


if __name__ == '__main__':
    unittest.main()
