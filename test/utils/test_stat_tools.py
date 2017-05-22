import unittest
from utils.bit_tools import least_significant_bit as lsb, byte_xor, split_chunks
from utils.stat_tools import standard_normal_distr_quantile, chi_squared_distr_quantile, \
    boxplot_params
from random import Random


class TestST(unittest.TestCase):
    def test_snd_quantile_precomputed(self):
        """Test if the function correctly returns the pre-computed quantiles"""
        values = [
            (0.0005, -3.2905), (0.001, -3.0902), (0.0025, -2.8070), (0.005, -2.5758),
            (0.01, -2.3263), (0.025, -1.9600), (0.05, -1.6449), (0.10, -1.2816), (0.15, -1.0364),
            (0.20, -0.8416), (0.25, -0.6745), (0.40, -0.2533), (0.60, 0.2533), (0.75, 0.6745),
            (0.80, 0.8416), (0.85, 1.0364), (0.90, 1.2816), (0.95, 1.6449), (0.975, 1.9600),
            (0.99, 2.3263), (0.995, 2.5758), (0.9975, 2.8070), (0.999, 3.0902), (0.9995, 3.2905)
        ]

        for v in values:
            self.assertEqual(standard_normal_distr_quantile(v[0]), v[1],
                             "Failed matching alpha=%d with x=%f" % (v[0], v[1]))

    def test_chi_sq_quantile_precomputed(self):
        """Test if the function correctly returns the pre-computed quantiles"""
        values = [
            (1, .9, 2.7055), (1, .95, 3.8415), (1, .995, 7.8794), (1, .999, 10.8276),
            (7, .9, 12.0170), (8, .95, 15.5073), (9, .990, 21.6660), (11, .999, 31.2641),
            (31, .9, 41.4217), (63, .95, 82.5287), (127, .975, 160.0858),
            (255, .99, 310.4574), (511, .995, 597.0978), (1023, .999, 1168.4972)

        ]

        for v in values:
            self.assertEqual(chi_squared_distr_quantile(v[1], dof=v[0]), v[2],
                             "Failed matching dof=%d, alpha=%d with x=%f" % (v[0], v[1], v[2]))

    def test_boxplot_params(self):
        """Test if the function correctly returns the pre-computed parameters."""
        data = [-2, -1, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
                6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 9, 9, 10, 11.5, 12, 13]

        params = boxplot_params(data)
        self.assertEqual(params['lq'], 4)
        self.assertEqual(params['med'], 5.5)
        self.assertEqual(params['uq'], 7)
        self.assertEqual(params['h'], 4.5)  # 1.5 * 3
        self.assertEqual(params['lf'], -0.5)
        self.assertEqual(params['uf'], 11.5)
        self.assertEqual(params['lw'], 0)
        self.assertEqual(params['uw'], 11.5)
        self.assertListEqual(params['out'], [-2, -1, 12, 13])


if __name__ == '__main__':
    unittest.main()
