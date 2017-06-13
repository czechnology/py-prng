from math import log2, sqrt, ceil
from sys import stderr

from scipy.special import erfc
from scipy.stats import norm

from utils.bit_tools import concat_chunks


def maurer_universal_test(generator, n_bits, l=None, q=None, k=None, sig_level=None, misc=None):
    """An implemementation of the Maurer's universal statistical test, as described by Menezes et
    al. [Men96]
    
    INPUT: a binary sequence s = s[0],s[1],...,s[n-1] of length n, and parameters L, Q, K.
    OUTPUT: the value of the statistic Xu for the sequence s.
    1.  (Zero the table T) For j from 0 to 2^L-1 do the following: T[j] = 0.
    2.  (Initialize the table T). For i from 1 to Q do the following: T[b[i]] = i.
    3.  sum = 0.
    4.  For i from Q+1 to Q+K do the following:
    4.1     sum = sum + log2(i - T[bi]).
    4.2     T[bi] = i.
    5. Xu = sum / K.
    6. Return(Xu).
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
    if not k:
        k = int(ceil(n_bits // l) - q)

    # print("n=%d, l=%d, q=%d, k=%d" % (n_bits, l, q, k))

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

    print("sum", sum_)

    xu = sum_ / k
    exp_mean, exp_var = maurer_mean_var2(l, k, c_alg=C_ALG_MAURER)
    zu = (xu - exp_mean) / sqrt(exp_var)

    erfc_arg = abs(xu - exp_mean) / sqrt(2) / sqrt(exp_var)
    p_value = erfc(erfc_arg)

    if type(misc) is dict:
        misc.update(sum_=sum_, xu=xu, zu=zu, p_value=p_value, l=l, q=q, k=k)

    if sig_level is None:
        return xu
    else:
        limit = -norm.ppf(sig_level / 2)
        if type(misc) is dict:
            misc.update(xu=xu, zu=zu, limit=limit, p_value=p_value)
        return -limit <= zu <= limit


C_ALG_MENEZES = 1
C_ALG_MAURER = 2


def maurer_mean_var2(l, k, c_alg=C_ALG_MENEZES):
    """Approximate mean and variance of the statistic Xu (in Maurer's test) for random sequences
    with parameters l,k as q -> Inf.
    """

    if not (1 <= l <= 16):
        raise ValueError("Out of bounds, l must be between 1 and 16")
    if k < 2 ** l:
        raise ValueError("Parameter k must be at least 2^l")

    mean_variance_1 = {
        1: (0.7326495, 0.690), 7: (6.1962507, 3.125), 13: (12.168070, 3.410),
        2: (1.5374383, 1.338), 8: (7.1836656, 3.238), 14: (13.167693, 3.416),
        3: (2.4016068, 1.901), 9: (8.1764248, 3.311), 15: (14.167488, 3.419),
        4: (3.3112247, 2.358), 10: (9.1723243, 3.356), 16: (15.167379, 3.421),
        5: (4.2534266, 2.705), 11: (10.170032, 3.384),
        6: (5.2177052, 2.954), 12: (11.168765, 3.401)
    }

    expected_value, expected_variance = mean_variance_1[l]

    if c_alg == C_ALG_MENEZES:
        # Algorithm as given by Menezes
        c_l_k = 0.7 - (0.8 / l) + (1.6 + (12.8 / l)) * (k ** (-4 / l))  # Menezes
        var2 = (c_l_k ** 2) * expected_variance / k
    elif c_alg == C_ALG_MAURER:
        # Algorithm as given by Maurer (and also used in NIST SP 800-22)
        c_l_k = 0.7 - (0.8 / l) + (4 + (32 / l)) * (k ** (-3 / l)) / 15  # Maurer
        var2 = (c_l_k ** 2) * expected_variance / k
    else:
        var2 = expected_variance

    return expected_value, var2
