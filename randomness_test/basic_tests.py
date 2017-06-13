from collections import deque
from math import log2, sqrt
from time import process_time as time

from scipy.special import erfc
from scipy.stats import chi2, norm

from generator.generator import StaticSequenceGenerator
from utils.bit_tools import concat_chunks
from utils.unit_tools import nicer_time


def frequency_test(generator, n_bits, sig_level=None, misc=None, n1=None):
    if n1 is None:
        n0, n1 = _calculate_n0_n1(generator, n_bits)
    else:
        n0 = n_bits - n1

    x1 = ((n0 - n1) ** 2) / n_bits
    p_value = erfc(sqrt(x1 / 2))

    if type(misc) is dict:
        misc.update(n0=n0, n1=n1, p_value=p_value)

    if sig_level is None:
        return x1
    else:
        limit = chi2.ppf(1 - sig_level, 1)
        if type(misc) is dict:
            misc.update(x=x1, limit=limit)
        return x1 <= limit


def _calculate_n0_n1(generator, n_bits):
    n1 = 0
    for _ in range(n_bits):
        n1 += generator.random_bit()
    n0 = n_bits - n1
    return n0, n1


def serial_test(generator, n_bits, n1=None, sig_level=None, misc=None):
    if n1 is None:
        n0, n1 = _calculate_n0_n1(generator, n_bits)
    else:
        n0 = n_bits - n1

    n_xx = [0] * 4  # number of occurrences of 00, 01, 10, 11
    bi0 = generator.random_bit()  # bit b[i]
    for i in range(n_bits - 1):
        bi1 = generator.random_bit()  # bit b[i+1]
        n_xx[bi0 << 1 | bi1] += 1
        bi0 = bi1

    # Calculate the statistic
    x2 = 4 / (n_bits - 1) * (n_xx[0b00] ** 2 + n_xx[0b01] ** 2 + n_xx[0b10] ** 2 + n_xx[0b11] ** 2)
    x2 += -2 / n_bits * (n0 ** 2 + n1 ** 2) + 1

    if type(misc) is dict:
        misc.update(n0=n0, n1=n1, n00=n_xx[0b00], n01=n_xx[0b01], n10=n_xx[0b10], n11=n_xx[0b11])

    if sig_level is None:
        return x2
    else:
        limit = chi2.ppf(1 - sig_level, 2)
        if type(misc) is dict:
            misc.update(x=x2, limit=limit)
        return x2 <= limit


def poker_test(generator, n_bits, m=None, sig_level=None, misc=None):
    if m is not None:
        if n_bits // m < 5 * (2 ** m):
            raise ValueError("Value m must satisfy requirement [n/m]>=5*2^m")
    else:
        # find the highest suitable m value
        m = int(log2(n_bits / 5))
        while n_bits // m < 5 * (2 ** m):
            m -= 1

    k = n_bits // m

    # Divide the sequence into k non-overlapping parts each of length m
    # and let ni be the number of occurrences of the ith type of sequence of length m, 1 <= i <= 2m.
    ni = [0] * (2 ** m)
    for i in range(0, k * m, m):
        t = concat_chunks([generator.random_bit() for _ in range(m)], bits=1)
        ni[t] += 1

    x3 = (2 ** m) / k * sum(map(lambda x: x ** 2, ni)) - k

    if type(misc) is dict:
        misc.update(m=m, k=k, ni=ni)

    if sig_level is None:
        return x3
    else:
        limit = chi2.ppf(1 - sig_level, (2 ** m) - 1)
        if type(misc) is dict:
            misc.update(x=x3, limit=limit)
        return x3 <= limit


def runs_test(generator, n_bits, sig_level=None, fips_style=False, misc=None):
    # The expected number of gaps (or blocks) of length i in a random sequence of length n is
    # e[i] = (n-i+3)=2^(i+2). Let k be equal to the largest integer i for which ei >= 5.
    def f_e(j):
        return (n_bits - j + 3) / 2 ** (j + 2)

    if fips_style:
        k = 6
    else:
        k = 1
        while f_e(k + 1) >= 5:
            k += 1

    # Let B[i], G[i] be the number of blocks and gaps, respectively, of length i in s for each i,
    # 1 <= i <= k.
    run_bit = None
    run_length = 0
    max_run_length = 0
    b = [0] * k  # zero-indexed
    g = [0] * k
    for i in range(n_bits + 1):
        bit = generator.random_bit() if i < n_bits else None

        # ongoing run
        if run_bit == bit:
            run_length += 1

        # run ended (or it's the beginning or end)
        if run_bit != bit:
            if run_length > max_run_length:
                max_run_length = run_length
            if fips_style and run_length > 6:
                run_length = 6
            if 1 <= run_length <= k:
                if run_bit == 0:
                    g[run_length - 1] += 1  # zero-indexed!
                elif run_bit == 1:
                    b[run_length - 1] += 1
            # reset counter
            run_bit = bit
            run_length = 1

    # Calculate the statistic:
    e = [f_e(i) for i in range(1, k + 1)]
    x4 = sum([(x - e[i]) ** 2 / e[i] for i, x in list(enumerate(b)) + list(enumerate(g))])

    if type(misc) is dict:
        misc.update(k=k, b=b, g=g, e=e, max_run=max_run_length)

    if sig_level is None:
        return x4
    else:
        limit = chi2.ppf(1 - sig_level, 2 * (k - 1))
        if type(misc) is dict:
            misc.update(x=x4, limit=limit)
        return x4 <= limit


def autocorrelation_test(generator, n_bits, d, sig_level=None, misc=None):
    if not (1 <= d <= n_bits // 2):
        raise ValueError("Parameter d must be between 1 and [n/2]=%d" % (n_bits // 2))

    # random bits from i to i+d
    generated_bits = deque([generator.random_bit() for _ in range(d)], maxlen=d)

    a = 0
    for i in range(n_bits - d):
        # a += sequence[i] ^ sequence[i + d]
        s_i_d = generator.random_bit()
        s_i_0 = generated_bits.popleft()
        generated_bits.append(s_i_d)
        a += s_i_0 ^ s_i_d

    # Calculate the statistic
    x5 = 2 * (a - (n_bits - d) / 2) / sqrt(n_bits - d)

    if type(misc) is dict:
        misc.update(a=a)

    if sig_level is None:
        return x5
    else:
        limit = -norm.ppf(sig_level / 2)
        if type(misc) is dict:
            misc.update(x=x5, limit=limit)
        return -limit <= x5 <= limit


def run_all(generator, n_bits, sig_level, continuous=False, print_log=False):
    # if we want all the tests to be applied to the *same* bit sequence,
    # we need to pre-compute it and create a static generator
    if not continuous:
        ts = time()
        sequence = generator.random_bytes((n_bits // 8) + 16)
        print(sequence)
        generator = StaticSequenceGenerator(seq=sequence)
        if print_log:
            print("(Sequence pre-computed in", nicer_time(time() - ts) + ')', flush=True)

    if not continuous:
        generator.rewind()  # rewind
    tf = frequency_test(generator, n_bits, sig_level=sig_level)

    if not continuous:
        generator.rewind()  # rewind
    ts = serial_test(generator, n_bits, sig_level=sig_level)

    if not continuous:
        generator.rewind()  # rewind
    tp = poker_test(generator, n_bits, sig_level=sig_level)

    if not continuous:
        generator.rewind()  # rewind
    tr = runs_test(generator, n_bits, sig_level=sig_level)

    if not continuous:
        generator.rewind()  # rewind
    tac = autocorrelation_test(generator, n_bits, d=100, sig_level=sig_level)

    return tf, ts, tp, tr, tac
