import unittest

from generator.linear_feedback_shift_registers import LinearFeedbackShiftRegister


class TestLinearFeedbackShiftRegisters(unittest.TestCase):
    SEQUENCES_PATH = "sequences/generalised_feedback_shift_registers"

    def test_lfsr_schneier_16_2(self):
        """Test against simple sequence as given by Schneier"""

        generator = LinearFeedbackShiftRegister(length=4, taps=(4, 1), seed=0b1111)
        expected_register = [0b1111, 0b0111, 0b1011, 0b0101, 0b1010, 0b1101, 0b0110, 0b0011, 0b1001,
                             0b0100, 0b0010, 0b0001, 0b1000, 0b1100, 0b1110]
        expected_sequence = [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

        for i in range(len(expected_register)):
            self.assertEqual(generator.register, expected_register[i])
            self.assertEqual(generator.random_bit(), expected_sequence[i])


if __name__ == '__main__':
    unittest.main()
