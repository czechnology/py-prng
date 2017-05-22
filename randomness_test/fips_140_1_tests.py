from randomness_test.basic_tests import frequency_test, poker_test, runs_test


def run_all(generator, seed=None):
    n = 20000

    if seed:
        generator.seed(seed)
    misc = {}
    frequency_test(generator, n, misc=misc)
    t1 = 9654 < misc['n1'] < 10346

    if seed:
        generator.seed(seed)
    x3 = poker_test(generator, n, m=4)
    t2 = 1.03 < x3 < 57.4

    if seed:
        generator.seed(seed)
    misc = {}
    runs_test(generator, n, fips_style=True, misc=misc)
    b, g, max_run = misc['b'], misc['g'], misc['max_run']
    t3 = all([2267 <= b[0] <= 2733, 2267 <= g[0] <= 2733,
              1079 <= b[1] <= 1421, 1079 <= g[1] <= 1421,
              502 <= b[2] <= 748, 502 <= g[2] <= 748,
              223 <= b[3] <= 402, 223 <= g[3] <= 402,
              90 <= b[4] <= 223, 90 <= g[4] <= 223,
              90 <= b[5] <= 223, 90 <= g[5] <= 223])

    t4 = max_run < 34

    return t1, t2, t3, t4
