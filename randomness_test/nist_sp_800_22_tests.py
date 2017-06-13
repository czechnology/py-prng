import operator
import statistics
import sys
from collections import deque, OrderedDict
from functools import reduce
from itertools import groupby, islice
from math import sqrt, log, log2, floor, ceil
from sys import stderr
from time import clock as time

import numpy as np
from scipy.fftpack import fft
from scipy.special import gammaincc as igamc, erfc
from scipy.stats import norm as ssn

from generator.generator import StaticSequenceGenerator
from randomness_test.compression_tests import maurer_mean_var2, C_ALG_MAURER
from utils.bit_tools import concat_chunks
from utils.unit_tools import nicer_time

"""
# References
[NIST10]Lawrence E Bassham III et al. A Statistical Test Suite for Random and Pseudorandom Number
        Generators for Cryptographic Applications. NIST Special Publication 800-22. Revision 1a.
        April 2010.
[Men96] Alfred J Menezes, Paul C Van Oorschot, and Vanstone. Handbook of applied cryptography,
        chapter 5. CRC press, 1996.
"""


def frequency(generator, n_bits, misc=None):
    """Frequency (Monobit) Test.
    
    Test purpose as described in [NIST10, section 2.1]:
    "The focus of the test is the proportion of zeroes and ones for the entire sequence. The purpose
    of this test is to determine whether the number of ones and zeros in a sequence are
    approximately the same as would be expected for a truly random sequence. The test assesses the
    closeness of the fraction of ones to 1/2, that is, the number of ones and zeroes in a sequence
    should be about the same. All subsequent tests depend on the passing of this test."
    """

    s_n = 0
    for _ in range(n_bits):
        s_n += 2 * generator.random_bit() - 1  # 1 if generator.random_bit() else -1

    s_obs = abs(s_n) / sqrt(n_bits)

    p_value = erfc(s_obs / sqrt(2))

    if type(misc) is dict:
        misc.update(s_n=s_n, s_obs=s_obs)

    return p_value


def block_frequency(generator, n_bits, m=None, misc=None):
    """Frequency Test within a Block.

    Test purpose as described in [NIST10, section 2.2]:
    "The focus of the test is the proportion of ones within M-bit blocks. The purpose of this test
    is to determine whether the frequency of ones in an M-bit block is approximately M/2, as would
    be expected under an assumption of randomness. For block size M=1, this test degenerates to
    test 1, the Frequency (Monobit) test."
    """
    if n_bits < 100:
        print("Warning: Sequence should be at least 100 bits long", file=stderr)
    if m:
        if m < 20 or m <= 0.01 * n_bits:
            print("Warning: Parameter m should satisfy m >= 20 and m > .01*n", file=stderr)
    else:
        m = max(20, int(0.01 * n_bits) + 1)

    n_blocks = n_bits // m
    assert (n_bits >= m * n_blocks)
    if n_blocks >= 100:
        print("Warning: Test should have less than 100 blocks", file=stderr)

    pi = []
    for i in range(n_blocks):
        pi.append(sum([generator.random_bit() for _ in range(m)]) / m)

    chi2_obs = 4 * m * sum(map(lambda p: (p - 0.5) ** 2, pi))
    p_value = igamc(n_blocks / 2, chi2_obs / 2)

    if type(misc) is dict:
        misc.update(m=m, n_blocks=n_blocks, pi=pi, chi2=chi2_obs)

    return p_value


def runs(generator, n_bits, misc=None):
    """Runs Test.

    Test purpose as described in [NIST10, section 2.3]:
    "The focus of this test is the total number of runs in the sequence, where a run is an
    uninterrupted sequence of identical bits. A run of length k consists of exactly k identical bits
    and is bounded before and after with a bit of the opposite value. The purpose of the runs test
    is to determine whether the number of runs of ones and zeros of various lengths is as expected
    for a random sequence. In particular, this test determines whether the oscillation between such
    zeros and ones is too fast or too slow."
    """
    pi = 0
    v_obs = 0
    b0 = None
    for _ in range(n_bits):
        b1 = generator.random_bit()
        pi += b1
        v_obs += b0 != b1
        b0 = b1
    pi /= n_bits

    if type(misc) is dict:
        misc.update(pi=pi, v_obs=v_obs)

    if abs(pi - 1 / 2) >= 2 / sqrt(n_bits):
        return 0

    p_value = erfc(abs(v_obs - 2 * n_bits * pi * (1 - pi)) / (2 * sqrt(2 * n_bits) * pi * (1 - pi)))

    return p_value


def longest_run_of_ones(generator, n_bits, m=None, misc=None):
    """Test for the Longest Run of Ones in a Block.

    Test purpose as described in [NIST10, section 2.4]:
    "The focus of the test is the longest run of ones within M-bit blocks. The purpose of this test
    is to determine whether the length of the longest run of ones within the tested sequence is
    consistent with the length of the longest run of ones that would be expected in a random
    sequence. Note that an irregularity in the expected length of the longest run of ones implies
    that there is also an irregularity in the expected length of the longest run of zeroes.
    Therefore, only a test for ones is necessary."
    """
    if not m:
        if n_bits >= 750000:
            m = 10000
        elif n_bits >= 6272:
            m = 128
        elif n_bits >= 128:
            m = 8
        else:
            raise ValueError("Sequence must be at least 128 bits long")
    elif m not in (8, 128, 10000):
        raise ValueError("Parameter m must be 8, 128 or 10000")

    if m == 8:
        min_run_length, max_run_length = 1, 4
        pi = (0.2148, 0.3672, 0.2305, 0.1875)
    elif m == 128:
        min_run_length, max_run_length = 4, 9
        pi = (0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124)
    elif m == 10000:
        min_run_length = 10
        max_run_length = 16
        pi = (0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727)
    else:
        raise Exception("This shouldn't happen")

    k = max_run_length - min_run_length
    n_blocks = n_bits // m

    v = [0] * (k + 1)
    for _ in range(n_blocks):
        block = [generator.random_bit() for _ in range(m)]
        runs_of_ones = [list(group) for k, group in groupby(block, bool) if k]
        if runs_of_ones:
            longest_run_length = max(map(len, runs_of_ones))
            # if run length is out of bounds, set it to the nearest bound
            # then the run length is the index (minus min_run_length because zero-indexed)
            index = min(max(longest_run_length, min_run_length), max_run_length) - min_run_length
            v[index] += 1

    chi2_obs = sum(map(lambda x: (x[0] - n_blocks * x[1]) ** 2 / (n_blocks * x[1]), zip(v, pi)))
    p_value = igamc(k / 2, chi2_obs / 2)

    if type(misc) is dict:
        misc.update(m=m, k=k, n=n_blocks, v=v, chi2_obs=chi2_obs)

    return p_value


def rank(generator, n_bits, misc=None):
    """Binary Matrix Rank Test.

    Test purpose as described in [NIST10, section 2.5]:
    "The focus of the test is the rank of disjoint sub-matrices of the entire sequence. The purpose
    of this test is to check for linear dependence among fixed length substrings of the original
    sequence. Note that this test also appears in the DIEHARD battery of tests."
    """
    m = q = 32
    n_blocks = n_bits // (m * q)

    if n_blocks < 1:
        raise ValueError("Sequence must be longer, should be at least 38*m*q =" +
                         str(38 * m * q) + "bits long")
    elif n_blocks < 38:
        print("Warning: Sequence should be at least 38*m*q =", 38 * m * q, "bits long", file=stderr)

    f_m = f_m1 = 0
    for _ in range(n_blocks):
        block = [generator.random_bit() for _ in range(m * q)]
        matrix = [block[i * q:(i + 1) * q] for i in range(m)]

        r = _matrix_rank_bin(matrix, m, q)
        if r == m:
            f_m += 1
        elif r == m - 1:
            f_m1 += 1

    # pre-calculate more precise values for probability values p1, p2, and p3
    # (see section 3.5 in NIST SP 800-22)
    p_ = 0
    p1 = 1
    # for x in range(1, 50): p1 *= 1 - (1.0 / (2 ** x))
    x = 1
    while abs(p1 - p_) > 1e-9:
        p_ = p1
        p1 *= 1 - (1.0 / (2 ** x))
        x += 1
    p2 = 2 * p1
    p3 = 1 - p1 - p2

    chi2_obs = (f_m - p1 * n_blocks) ** 2 / p1 / n_blocks + \
               (f_m1 - p2 * n_blocks) ** 2 / p2 / n_blocks + \
               (n_blocks - f_m - f_m1 - p3 * n_blocks) ** 2 / p3 / n_blocks
    # p_value = exp(-chi2_obs/2)
    p_value = igamc(1, chi2_obs / 2)

    if type(misc) is dict:
        misc.update(n=n_blocks, f_m=f_m, f_m1=f_m1, chi2_obs=chi2_obs)

    return p_value


def discrete_fourier_transform(generator, n_bits, misc=None):
    """Discrete Fourier Transform (Spectral) Test.

    Test purpose as described in [NIST10, section 2.6]:
    "The focus of this test is the peak heights in the Discrete Fourier Transform of the sequence.
    The purpose of this test is to detect periodic features (i.e., repetitive patterns that are near
    each other) in the tested sequence that would indicate a deviation from the assumption of
    randomness. The intention is to detect whether the number of peaks exceeding the 95 % threshold
    is significantly different than 5 %."
    """
    if n_bits < 1000:
        print("Warning: Sequence should be at least 1000 bits long", file=stderr)

    x = [1 if generator.random_bit() else -1 for _ in range(n_bits)]
    s = fft(x)  # S = DFT(X)
    # print(s)
    m = abs(s)[0:n_bits // 2]  # modulus(S')
    # print("m =", m)
    t = sqrt(log(1 / 0.05) * n_bits)  # the 95% peak height threshold value
    n_0 = 0.95 * n_bits / 2  # expected theoretical number of peaks under t
    n_1 = len([1 for p in m if p < t])
    d = (n_1 - n_0) / sqrt(n_bits * 0.95 * 0.05 / 4)
    p_value = erfc(abs(d) / sqrt(2))

    if type(misc) == dict:
        misc.update(m=m, t=t, n_0=n_0, n_1=n_1, d=d)

    return p_value


def non_overlapping_template_matching(generator, n_bits, b=None, n_blocks=None, misc=None):
    """Non-overlapping Template Matching Test.

    Test purpose as described in [NIST10, section 2.7]:
    "The focus of this test is the number of occurrences of pre-specified target strings. The
    purpose of this test is to detect generators that produce too many occurrences of a given
    non-periodic (aperiodic) pattern. For this test and for the Overlapping Template Matching test
    of Section 2.8, an m-bit window is used to search for a specific m-bit pattern. If the pattern
    is not found, the window slides one bit position. If the pattern is found, the window is reset
    to the bit after the found pattern, and the search resumes."
    """
    if not b:
        b = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif type(b) == tuple:
        b = list(b)
    b_len = len(b)

    if not n_blocks:
        n_blocks = 8
    m = n_bits // n_blocks

    exp_mean = (m - b_len + 1) / (2 ** b_len)
    exp_var = m * (1 / (2 ** b_len) - (2 * b_len - 1) / (2 ** (2 * b_len)))

    n_blocks = n_bits // m
    chi2_obs = 0
    for _ in range(n_blocks):
        block = [generator.random_bit() for _ in range(m)]
        i, w = 0, 0
        while i < m - b_len + 1:
            if block[i:i + b_len] == b:
                w += 1
                i += b_len
            else:
                i += 1
        chi2_obs += ((w - exp_mean) ** 2) / exp_var

    p_value = igamc(n_blocks / 2, chi2_obs / 2)

    if type(misc) == dict:
        misc.update(exp_mean=exp_mean, exp_var=exp_var, n=n_blocks, chi2_obs=chi2_obs)

    return p_value


def overlapping_template_matching(generator, n_bits, b=(1, 1, 1, 1, 1, 1, 1, 1, 1),
                                  misc=None, use_old_nist_pi=False):
    """Overlapping Template Matching Test.

    Test purpose as described in [NIST10, section 2.8]:
    "The focus of the Overlapping Template Matching test is the number of occurrences of
    pre-specified target strings. Both this test and the Non-overlapping Template Matching test of
    Section 2.7 use an m-bit window to search for a specific m-bit pattern. As with the test in
    Section 2.7, if the pattern is not found, the window slides one bit position. The difference
    between this test and the test in Section 2.7 is that when the pattern is found, the window
    slides only one bit before resuming the search."
    """

    if type(b) == tuple:
        b = list(b)
    b_len = len(b)
    # if b_len != 9:
    #     raise ValueError("Length of pattern b must be exactly 9 bits")

    m = 1032  # hardcoded to match the pre-computed pi values that depend on m and length of b
    n_blocks = n_bits // m

    lambda_ = (m - b_len + 1) / (2 ** b_len)
    eta = lambda_ / 2

    v = [0] * 6
    for _ in range(n_blocks):
        block = [generator.random_bit() for _ in range(m)]
        w = 0
        for i in range(m - b_len + 1):
            if block[i:i + b_len] == b:
                w += 1
        v[min(w, 5)] += 1

    if use_old_nist_pi:
        pi = (0.367879, 0.183940, 0.137955, 0.099634, 0.069935, 0.140657)
    else:  # use improved pi values given by Hamano and Kaneko
        pi = (0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865)

    chi2_obs = sum(map(lambda x: (x[0] - n_blocks * x[1]) ** 2 / (n_blocks * x[1]), zip(v, pi)))

    p_value = igamc(5 / 2, chi2_obs / 2)

    if type(misc) == dict:
        misc.update(v=v, lambda_=lambda_, eta=eta, chi2_obs=chi2_obs)

    return p_value


def universal(generator, n_bits, l=None, q=None, misc=None):
    """Maurer's "Universal Statistical" Test.

    Test purpose as described in [NIST10, section 2.9]:
    "The focus of this test is the number of bits between matching patterns (a measure that is
    related to the length of a compressed sequence). The purpose of the test is to detect whether 
    or not the sequence can be significantly compressed without loss of information. A significantly
    compressible sequence is considered to be non-random."
    """
    if not l:
        if n_bits >= 1059061760:
            l = 16
        elif n_bits >= 496435200:
            l = 15
        elif n_bits >= 231669760:
            l = 14
        elif n_bits >= 107560960:
            l = 13
        elif n_bits >= 49643520:
            l = 12
        elif n_bits >= 22753280:
            l = 11
        elif n_bits >= 10342400:
            l = 10
        elif n_bits >= 4654080:
            l = 9
        elif n_bits >= 2068480:
            l = 8
        elif n_bits >= 904960:
            l = 7
        elif n_bits >= 387840:
            l = 6
        else:
            l = 5

    if not q:
        q = 10 << l

    k = n_bits // l - q

    # verify parameters
    if q < (10 << l):
        print("Warning: Parameter q should be at least 10 * 2^L", file=stderr)
    # if k < 1000 * (2 ** l):
    #     raise ValueError("Parameter k should be at least 1000 * 2^L = " + str(1000 * (2 ** l)))
    if q + k != n_bits // l:
        raise ValueError("Parameters q+k must be equal to the number of l-bit blocks in sequence")
    if n_bits < (q + k) * l:
        raise ValueError("The sequence should be at least 1010 * 2^L * l = %d bits long"
                         % ((q + k) * l))

    b = []
    for i in range(0, n_bits, l):
        b.append(concat_chunks([generator.random_bit() for _ in range(l)], bits=1))

    t = [0] * (2 ** l)
    for i in range(1, q + 1):
        t[b[i - 1]] = i

    sum_ = 0
    for i in range(q + 1, q + k + 1):
        sum_ += log2(i - t[b[i - 1]])
        t[b[i - 1]] = i

    fn = sum_ / k

    exp_mean, exp_var = maurer_mean_var2(l, k, c_alg=C_ALG_MAURER)
    erfc_arg = abs(fn - exp_mean) / sqrt(2) / sqrt(exp_var)
    p_value = erfc(erfc_arg)

    if type(misc) is dict:
        misc.update(t=t, sum_=sum_, fn=fn, l=l, q=q, k=k, erfc_arg=erfc_arg)

    return p_value


def linear_complexity(generator, n_bits, m=500, misc=None, use_old_nist_pi=False):
    """Linear Complexity Test.

    Test purpose as described in [NIST10, section 2.10]:
    "The focus of this test is the length of a linear feedback shift register (LFSR). The purpose of
    this test is to determine whether or not the sequence is complex enough to be considered random.
    Random sequences are characterized by longer LFSRs. An LFSR that is too short implies
    non-randomness."
    """

    n_blocks = n_bits // m

    if n_bits < 10 ** 6:
        print("Warning: Sequence should be at least 10^6 bits long", file=stderr)
    if not (500 <= m < 5000):
        print("Warning: Parameter m should be in range [500, 5000]", file=stderr)
    if n_blocks < 200:
        print("Warning: The sequence should be split to at least 200 blocks", file=stderr)

    exp_mean = m / 2 + (9 + (-1) ** (m + 1)) / 36 - (m / 3 + 2 / 9) / (2 ** m)

    v = [0] * 7
    for i in range(n_blocks):
        block = [generator.random_bit() for _ in range(m)]
        l = _berlekamp_massey(block)
        t = ((-1) ** m) * (l - exp_mean) + 2 / 9
        index = int(min(max(ceil(t + 2.5), 0), 6))
        v[index] += 1

    if use_old_nist_pi:
        # old NIST values - these pass the test samples but the first value is incorrect
        pi = (0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833)
    else:
        # correct pi values
        pi = (1 / 96, 0.03125, 0.125, 0.5, 0.25, 0.0625, 1 / 48)

    chi2_obs = sum(map(lambda v_i, pi_i: (v_i - n_blocks * pi_i) ** 2 / (n_blocks * pi_i), v, pi))
    p_value = igamc(6 / 2., chi2_obs / 2.)

    if type(misc) == dict:
        misc.update(exp_mean=exp_mean, v=v, chi2_obs=chi2_obs)

    return p_value


def serial(generator, n_bits, m=16, misc=None):
    """Serial Test.

    Test purpose as described in [NIST10, section 2.11]:
    "The focus of this test is the frequency of all possible overlapping m-bit patterns across the
    entire sequence. The purpose of this test is to determine whether the number of occurrences of
    the 2m m-bit overlapping patterns is approximately the same as would be expected for a random
    sequence. Random sequences have uniformity; that is, every m-bit pattern has the same chance of
    appearing as every other m-bit pattern. Note that for m = 1, the Serial test is equivalent to
    the Frequency test of Section 2.1."
    """

    if m < 1:
        raise ValueError("Parameter m must be an integer greater than zero")

    if m >= int(log2(n_bits)) - 2:
        print("Warning: Parameter m should be less than [log2(n)]-2", file=stderr)

    sequence_start = [generator.random_bit() for _ in range(m - 1)]
    window = deque(sequence_start, maxlen=m)
    vm0 = [0] * (2 ** m)
    vm1 = [0] * (2 ** (m - 1)) if m >= 2 else []
    vm2 = [0] * (2 ** (m - 2)) if m >= 3 else []
    for i in range(m - 1, n_bits + (m - 1)):
        bit = generator.random_bit() if i < n_bits else sequence_start[i - n_bits]
        window.append(bit)  # when full with m items, left item is discarded

        vm0[concat_chunks(window, bits=1)] += 1
        if m >= 2:
            vm1[concat_chunks(islice(window, 0, m - 1), bits=1)] += 1
        if m >= 3:
            vm2[concat_chunks(islice(window, 0, m - 2), bits=1)] += 1

    psi2m0 = (2 ** m) / n_bits * sum(map(lambda x: x ** 2, vm0)) - n_bits
    psi2m1 = (2 ** (m - 1)) / n_bits * sum(map(lambda x: x ** 2, vm1)) - n_bits if vm1 else 0
    psi2m2 = (2 ** (m - 2)) / n_bits * sum(map(lambda x: x ** 2, vm2)) - n_bits if vm2 else 0

    delta_psi2m = psi2m0 - psi2m1
    delta2_psi2m = psi2m0 - 2 * psi2m1 + psi2m2

    p_value1 = igamc(2 ** (m - 2), delta_psi2m / 2)
    p_value2 = igamc(2 ** (m - 3), delta2_psi2m / 2)

    if type(misc) == dict:
        misc.update(vm0=vm0, vm1=vm1, vm2=vm2, psi2m0=psi2m0, psi2m1=psi2m1, psi2m2=psi2m2,
                    delta_psi2m=delta_psi2m, delta2_psi2m=delta2_psi2m)

    return p_value1, p_value2


def approximate_entropy(generator, n_bits, m=10, misc=None):
    """Approximate Entropy Test.

    Test purpose as described in [NIST10, section 2.12]:
    "As with the Serial test of Section 2.11, the focus of this test is the frequency of all
    possible overlapping m-bit patterns across the entire sequence. The purpose of the test is to
    compare the frequency of overlapping blocks of two consecutive/adjacent lengths (m and m+1)
    against the expected result for a random sequence."
    """

    if m < 1:
        raise ValueError("Parameter m must be an integer greater than zero")

    if m >= int(log2(n_bits)) - 5:
        print("Warning: Parameter m should be less than [log2(n)]-5", file=stderr)

    vm0 = [0] * (2 ** m)
    vm1 = [0] * (2 ** (m + 1))

    sequence_start = [generator.random_bit() for _ in range(m)]
    window0 = deque(sequence_start, maxlen=m)
    window1 = deque(sequence_start, maxlen=m + 1)

    # take care of the first step for the smaller window
    vm0[concat_chunks(sequence_start, bits=1)] += 1

    all_seq = sequence_start
    for i in range(m, n_bits + m):
        bit = generator.random_bit() if i < n_bits else sequence_start[i - n_bits]
        all_seq.append(bit)
        window0.append(bit)  # when full with m items, left item is discarded
        window1.append(bit)  # when full with m+1 items, left item is discarded

        if i < n_bits + m - 1:  # skip the last step for the smaller window
            vm0[concat_chunks(window0, bits=1)] += 1
        vm1[concat_chunks(window1, bits=1)] += 1

    cm0 = map(lambda x: x / n_bits, vm0)
    cm1 = map(lambda x: x / n_bits, vm1)

    phi_m0 = sum([x * log(x) for x in cm0 if x > 0])
    phi_m1 = sum([x * log(x) for x in cm1 if x > 0])

    chi2_obs = 2 * n_bits * (log(2) - (phi_m0 - phi_m1))
    p_value = igamc(2 ** (m - 1), chi2_obs / 2)

    if type(misc) == dict:
        misc.update(vm0=vm0, vm1=vm1, phi_m0=phi_m0, phi_m1=phi_m1, chi2_obs=chi2_obs)

    return p_value


def cumulative_sums(generator, n_bits, forward_mode=True, misc=None):
    """Approximate Entropy Test.

    Test purpose as described in [NIST10, section 2.13]:
    "The focus of this test is the maximal excursion (from zero) of the random walk defined by the
    cumulative sum of adjusted (-1, +1) digits in the sequence. The purpose of the test is to
    determine whether the cumulative sum of the partial sequences occurring in the tested sequence
    is too large or too small relative to the expected behavior of that cumulative sum for random
    sequences. This cumulative sum may be considered as a random walk. For a random sequence, the
    excursions of the random walk should be near zero. For certain types of non-random sequences,
    the excursions of this random walk from zero will be large."
    """

    if n_bits < 100:
        print("Warning: Sequence should be at least 100 bits long", file=stderr)

    s = [0] * n_bits
    if forward_mode:
        # forward mode is simpler and allows to collect the cumulative sum "on the go"
        for i in range(n_bits):
            x = 1 if generator.random_bit() else - 1
            s[i] = s[i - 1] + x
    else:
        # backward mode requires pre-loading the whole sequence or nested loops
        sequence = [1 if generator.random_bit() else - 1 for _ in range(n_bits)]
        for i, x in enumerate(reversed(sequence)):
            s[i] = s[i - 1] + x

    z = max(map(abs, s))

    def phi(k, c):
        return ssn.cdf((4 * k + c) * z / sqrt(n_bits))

    # the intervals for the sums
    int1 = map(floor, np.arange((-n_bits / z + 1) / 4, (n_bits / z - 1) / 4))
    int2 = map(floor, np.arange((-n_bits / z - 3) / 4, (n_bits / z - 1) / 4))

    sum1 = sum(map(lambda k: phi(k, 1) - phi(k, -1), int1))
    sum2 = sum(map(lambda k: phi(k, 3) - phi(k, 1), int2))
    p_value = 1 - sum1 + sum2

    if type(misc) == dict:
        misc.update(cumsum=s, z=z)

    return p_value


def cumulative_sums_forward(generator, n_bits, misc=None):
    """An convenience function to the cumulative_sums test with forward mode on."""
    return cumulative_sums(generator, n_bits, forward_mode=True, misc=misc)


def cumulative_sums_backward(generator, n_bits, misc=None):
    """An convenience function to the cumulative_sums test with backward mode on."""
    return cumulative_sums(generator, n_bits, forward_mode=False, misc=misc)


def random_excursions(generator, n_bits, misc=None):
    """Random Excursions Test.

    Test purpose as described in [NIST10, section 2.14]:
    "The focus of this test is the number of cycles having exactly K visits in a cumulative sum
    random walk. The cumulative sum random walk is derived from partial sums after the (0,1)
    sequence is transferred to the appropriate (-1, +1) sequence. A cycle of a random walk consists
    of a sequence of steps of unit length taken at random that begin at and return to the origin.
    The purpose of this test is to determine if the number of visits to a particular state within a
    cycle deviates from what one would expect for a random sequence. This test is actually a series
    of eight tests (and conclusions), one test and conclusion for each of the states: -4, -3, -2, -1
    and +1, +2, +3, +4."
    """

    if n_bits < 10 ** 6:
        print("Warning: Sequence should be at least 10^6 bits long", file=stderr)

    s = [0] * n_bits
    for i in range(n_bits):
        x = 1 if generator.random_bit() else - 1
        s[i] = s[i - 1] + x
    s.append(0)  # leading zero not needed for our implementation

    v = [[0] * 6 for _ in range(8)]
    f = [0] * 8
    j = 0
    for x in s:
        if x == 0:
            j += 1
            for i, y in enumerate(f):
                v[i][min(y, 5)] += 1
            f = [0] * 8
        elif -4 <= x <= 4:
            f[x + 4 - (x > 0)] += 1

    if j < 500:
        print("Warning: The number of cycles (zero crossings) should be at least 500 (is %d)" % j,
              file=stderr)

    def pi(k, x):
        a = 1 - 1 / 2 / abs(x)
        if k == 0:
            return a
        elif k < 5:
            return 1 / 4 / (x ** 2) * (a ** (k - 1))
        else:  # k >= 5
            return 1 / 2 / abs(x) * (a ** 4)

    chi2_obs = []
    for i, x in enumerate(list(range(-4, 0)) + list(range(1, 5))):
        chi2_obs.append(sum(map(lambda v_k_x, pi_k_x: (v_k_x - j * pi_k_x) ** 2 / j / pi_k_x,
                                v[i], [pi(k, x) for k in range(0, 6)])))

    p_value = list(map(lambda chi: igamc(5 / 2, chi / 2), chi2_obs))

    if type(misc) == dict:
        misc.update(cumsum=s, j=j, v=v, chi2_obs=chi2_obs)

    return p_value


def random_excursions_variant(generator, n_bits, misc=None):
    """Random Excursions Variant Test.

    Test purpose as described in [NIST10, section 2.15]:
    "The focus of this test is the total number of times that a particular state is visited (i.e.,
    occurs) in a cumulative sum random walk. The purpose of this test is to detect deviations from
    the expected number of visits to various states in the random walk. This test is actually a
    series of eighteen tests (and conclusions), one test and conclusion for each of the states:
    -9, -8, ..., -1 and +1, +2, ..., +9."
    """

    if n_bits < 10 ** 6:
        print("Warning: Sequence should be at least 10^6 bits long", file=stderr)

    s = [0] * n_bits
    for i in range(n_bits):
        x = 1 if generator.random_bit() else - 1
        s[i] = s[i - 1] + x
    s.append(0)  # leading zero not needed for our implementation

    ksi = [0] * 18
    j = 0
    for x in s:
        if x == 0:
            j += 1
        elif -9 <= x <= 9:
            ksi[x + 9 - (x > 0)] += 1

    if j < 500:
        print("Warning: The number of cycles (zero crossings) should be at least 500 (is %d)" % j,
              file=stderr)

    p_value = list(map(lambda ksi_i, x: erfc(abs(ksi_i - j) / sqrt(2 * j * (4 * abs(x) - 2))),
                       ksi, list(range(-9, 0)) + list(range(1, 10))))

    if type(misc) == dict:
        misc.update(cumsum=s, j=j, ksi=ksi)

    return p_value


all_tests = OrderedDict((
    ('frequency', 'Frequency (Monobit) Test'),
    ('block_frequency', 'Frequency Test within a Block'),
    ('runs', 'Runs Test'),
    ('longest_run_of_ones', 'Test for the Longest-Run-of-Ones in a Block'),
    ('rank', 'Binary Matrix Rank Test'),
    ('discrete_fourier_transform', 'Discrete Fourier Transform (Spectral) Test'),
    ('non_overlapping_template_matching', 'Non-overlapping Template Matching Test'),
    ('overlapping_template_matching', 'Overlapping Template Matching Test'),
    ('universal', 'Maurer\'s "Universal Statistical" Test'),
    ('linear_complexity', 'Linear Complexity Test'),
    ('serial', 'Serial Test'),
    ('approximate_entropy', 'Approximate Entropy Test'),
    ('cumulative_sums_forward', 'Forward Cumulative Sums Test'),
    ('cumulative_sums_backward', 'Backward Cumulative Sums Test'),
    ('random_excursions', 'Random Excursions Test'),
    ('random_excursions_variant', 'Random Excursions Variant Test'),
))


def run_all(generator, n_bits, continuous=False, print_log=False):
    results = []
    ts_total = time()

    # if we want all the tests to be applied to the *same* bit sequence,
    # we need to pre-compute it and create a static generator
    if not continuous:
        ts = time()
        sequence = generator.random_bytes((n_bits + 1024) // 8)  # 1024 extra bits
        generator = StaticSequenceGenerator(seq=sequence)
        if print_log:
            print("(Sequence pre-computed in", nicer_time(time() - ts) + ')', flush=True)

    for t_id, t_name in all_tests.items():
        if not continuous:
            # print("(Rewinding sequence before next test)")
            generator.rewind()  # rewind

        test = globals()[t_id]
        ts = time()
        p_value = test(generator, n_bits)
        te = time() - ts
        results.append((p_value, t_id, t_name))
        if print_log:
            sys.stderr.flush()
            if type(p_value) in (tuple, list):
                print("P-value: %.4f avg\t(%s, %.3f sec)"
                      % (statistics.mean(p_value), t_name, te), flush=True)
            else:
                print("P-value: %.4f\t\t(%s, %.3f sec)" % (p_value, t_name, te), flush=True)
    print("Finished NIST tests in %.1f seconds" % (time() - ts_total))

    # t1 = frequency(generator, n_bits)
    # t2 = block_frequency(generator, n_bits)
    # t3 = runs(generator, n_bits)
    # t4 = longest_run_of_ones(generator, n_bits)
    # t5 = rank(generator, n_bits)
    # t6 = discrete_fourier_transform(generator, n_bits)
    # t7 = non_overlapping_template_matching(generator, n_bits)
    # t8 = overlapping_template_matching(generator, n_bits)
    # t9 = universal(generator, n_bits)
    # t10 = linear_complexity(generator, n_bits)
    # t11 = serial(generator, n_bits)
    # t12 = approximate_entropy(generator, n_bits)
    # t13fw = cumulative_sums_forward(generator, n_bits)
    # t13bw = cumulative_sums_backward(generator, n_bits)
    # t14 = random_excursions(generator, n_bits)
    # t15 = random_excursions_variant(generator, n_bits)
    # return t1, t2, t3, t4

    return results


def _matrix_rank_bin(matrix, m, q):
    def apply_elementary_row_operations():
        for i in range(m):
            if matrix[i][i] == 0:
                for k in range(i + 1, q):
                    if matrix[k][i] == 1:
                        matrix[i], matrix[k] = matrix[k], matrix[i]
                        break
            for k in range(i + 1, q):
                if matrix[k][i] == 1:
                    matrix[k][i:] = list(map(lambda p: p[0] ^ p[1],
                                             zip(matrix[k][i:], matrix[i][i:])))
        return matrix

    # forward application of elementary row operations
    apply_elementary_row_operations()

    # transpose matrix
    matrix = [row[::-1] for row in reversed(matrix)]

    # backward application of elementary row operations
    apply_elementary_row_operations()

    return sum(map(any, matrix))


def _berlekamp_massey(s):
    """The Berlekamp-Massey algorithm for determining the linear complexity of a finite binary
    sequence. Implemented as defined in [Men97], chapter 6.2.3"""
    # Probably easier to understand from Wikipedia:
    # https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Massey_algorithm#The_algorithm_for_the_binary_field

    b, c = [0] * len(s), [0] * len(s)
    b[0] = c[0] = 1
    l, m = 0, -1

    for n in range(len(s)):
        d = reduce(operator.xor, map(operator.and_, c[1:l + 1], reversed(s[n - l:n])), s[n])
        if d == 1:
            t = list(c)  # for some strange reason, casting to list is needed here
            c[n - m:len(s)] = map(operator.xor, c[n - m:len(s)], b[0:len(s) - n + m])
            if float(l) <= n / 2:
                l = n + 1 - l
                m = n
                b = t

    return l
