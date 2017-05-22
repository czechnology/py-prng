import unittest
from utils.bit_tools import least_significant_bit as lsb, byte_xor, split_chunks, concat_chunks, \
    bits_to_byte, byte_to_bits, eliminate_bias
from random import Random


class TestBT(unittest.TestCase):
    def test_least_significant_bit_basic(self):
        """Test if the function correctly returns the least significant bit(s)"""

        number = 0b10110000111

        self.assertEqual(lsb(number), 0b1)
        self.assertEqual(lsb(number, 4), 0b0111)
        self.assertEqual(lsb(number, 9), 0b110000111)
        self.assertEqual(lsb(number, 10), 0b110000111)
        self.assertEqual(lsb(number, 11), 0b10110000111)

    def test_least_significant_bit_long(self):
        """Test if the function correctly returns the least significant bit(s) from long integers"""

        number = 0b111110010001000101001101010101001111101010011010000111011110010010011110011011101
        self.assertEqual(lsb(number), 0b1)

        number = 0b101010110001010110101010101111100101101000001101001010110001101000101101010101010
        self.assertEqual(lsb(number), 0b0)

    def test_byte_xor(self):
        ba1 = [0b10101010, 0b11111111, 0b00000000]
        ba2 = [0b01010101, 0b11111111, 0b10101010]
        xor = [0b11111111, 0b00000000, 0b10101010]
        self.assertEqual(byte_xor(ba1, ba2), xor)

    def test_split_chunks(self):
        r = Random(1)

        number = r.getrandbits(128)
        chunks = split_chunks(number, 32)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(
            '{:0128b}'.format(number),
            ''.join(map(lambda c: '{:032b}'.format(c), chunks)))

        number = r.getrandbits(512)
        chunks = split_chunks(number, 32)
        self.assertEqual(len(chunks), 16)
        self.assertEqual(
            '{:0512b}'.format(number),
            ''.join(map(lambda c: '{:032b}'.format(c), chunks)))

        chunks = split_chunks(number, 64)
        self.assertEqual(len(chunks), 8)
        self.assertEqual(
            '{:0512b}'.format(number),
            ''.join(map(lambda c: '{:064b}'.format(c), chunks)))

        number = r.getrandbits(513)
        chunks = split_chunks(number, 64)
        self.assertEqual(len(chunks), 9)
        self.assertEqual(
            '{:0576b}'.format(number),
            ''.join(map(lambda c: '{:064b}'.format(c), chunks)))

        number = r.getrandbits(64)
        chunks = split_chunks(number, 32, pad=4)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(
            '{:0128b}'.format(number),
            ''.join(map(lambda c: '{:032b}'.format(c), chunks)))

        chunks = split_chunks(number, 1)
        self.assertEqual(len(chunks), number.bit_length())
        self.assertEqual(
            '{:b}'.format(number),
            ''.join(map(lambda c: '{:b}'.format(c), chunks)))
        self.assertTrue(all(map(lambda n: 0 <= n <= 1, chunks)))

        self.assertEqual(split_chunks(0b0110, 1, pad=4), [0, 1, 1, 0])

    def test_concat_chunks(self):
        chunks = [0b0001, 0b1010, 0b0010]
        self.assertEqual(concat_chunks(chunks, bits=4), 0b110100010)
        self.assertEqual(concat_chunks(chunks, bits=5), 0b10101000010)

    def test_bits_to_byte(self):
        self.assertEqual(bits_to_byte([0, 0, 1, 0, 1, 1, 0, 1]), 0b00101101)
        self.assertEqual(bits_to_byte([1, 0, 0, 0, 1, 0, 1, 1]), 0b10001011)
        self.assertEqual(bits_to_byte([1, 1, 1, 1, 1, 1, 1, 1]), 255)
        self.assertEqual(bits_to_byte([0, 0, 0, 0, 0, 0, 0, 0]), 0)
        with self.assertRaises(ValueError):
            bits_to_byte([0, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            bits_to_byte([0, 0, 0, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            bits_to_byte([0, 0, 0, 0, 2, 0, 0, 0])

    def test_byte_to_bits(self):
        self.assertEqual(byte_to_bits(0b00101101), [0, 0, 1, 0, 1, 1, 0, 1])
        self.assertEqual(byte_to_bits(0b10001011), [1, 0, 0, 0, 1, 0, 1, 1])
        self.assertEqual(byte_to_bits(255), [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(byte_to_bits(0), [0, 0, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            byte_to_bits(-1)
        with self.assertRaises(ValueError):
            byte_to_bits(256)

    def test_eliminate_bias(self):
        seq = [0b00000010, 0b00001000, 0b00100001, 0b01000010, 0b00010100, 0b00010000, 0b01001000]

        seq_unbiased = eliminate_bias(seq)
        self.assertEqual(seq_unbiased, [0b11100100])  # dropped: 001

        seq.extend([0b01001011, 0b10011000])

        seq_unbiased = eliminate_bias(seq)
        self.assertEqual(seq_unbiased, [0b11100100, 0b00101101])  # dropped: 001


if __name__ == '__main__':
    unittest.main()
