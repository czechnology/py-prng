from math import gcd, copysign, exp, gamma, factorial


def is_power2(num):
    """Check if a number is a power of two, i.e. num = 2**e for some positive integer e"""
    # based on http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/

    return (num & (num - 1)) == 0 and num > 0


def lcm(a, b):
    """Return lowest common multiple of integers a and b."""
    # from http://stackoverflow.com/a/147539
    return a * b // gcd(a, b)


def erf_approx(x):
    # https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
    # Numerical approximation with a maximal error of 1.2e-7

    t = 1 / (1 + abs(x) / 2)

    tau = t * exp(-x * x - 1.26551223 + 1.00002368 * t
                  + 0.37409196 * (t ** 2) + 0.09678418 * (t ** 3)
                  - 0.18628806 * (t ** 4) + 0.27886807 * (t ** 5)
                  - 1.13520398 * (t ** 6) + 1.48851587 * (t ** 7)
                  - 0.82215223 * (t ** 8) + 0.17087277 * (t ** 9))

    return copysign(1 - tau, x)


def erfc_approx(x):
    return 1 - erf_approx(x)


def incomplete_gamma_lower(a, x, precision=9):
    # Numerical approximation of the lower incomplete gamma function to the given precision
    # https://en.wikipedia.org/wiki/Incomplete_gamma_function
    # http://mathworld.wolfram.com/IncompleteGammaFunction.html

    max_diff = 10 ** -precision

    def summand(k):
        return ((-x) ** k) / factorial(k) / (a + k)

    xs = x ** a
    sum_ = summand(0)
    prev_value = xs * sum_  # for k=0
    sum_ += summand(1)
    cur_value = xs * sum_  # for k=1
    k = 1
    while abs(cur_value - prev_value) > max_diff:
        k += 1
        prev_value = cur_value
        sum_ += summand(k)
        cur_value = xs * sum_

    return cur_value


def incomplete_gamma_upper(a, x, precision=9):
    # Numerical approximation of the lower incomplete gamma function to the given precision
    # https://en.wikipedia.org/wiki/Incomplete_gamma_function
    # http://mathworld.wolfram.com/IncompleteGammaFunction.html
    return gamma(a) - incomplete_gamma_lower(a, x, precision)


def incomplete_gamma_lower_norm(a, x, precision=9):
    return incomplete_gamma_lower(a, x, precision) / gamma(a)


def incomplete_gamma_upper_norm(a, x, precision=9):
    return incomplete_gamma_upper(a, x, precision) / gamma(a)
