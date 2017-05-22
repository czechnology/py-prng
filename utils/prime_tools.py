from random import Random

TEST_FERMAT = 1


def is_prime(n, test=TEST_FERMAT):
    if test == TEST_FERMAT:
        return _is_prime_fermat(n)
    else:
        raise ValueError("Unknown test")


def _is_prime_fermat(n, t=10):
    # 1.    For i from 1 to t do the following:
    # 1.1       Choose a random integer a, 2 <= a <= n-2.
    # 1.2       Compute r=a^(n−1) mod n     (using Algorithm 2.143).
    # 1.3       If r!=1 then return("composite").
    # 2.    Return("prime")

    if n == 2 or n == 3:
        return True

    rand = Random()

    for _ in range(t):
        a = rand.randint(2, n - 2)
        r = pow(a, n - 1, n)
        if r != 1:
            return False

    return True
