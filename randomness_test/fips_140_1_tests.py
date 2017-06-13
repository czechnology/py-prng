from time import process_time as time

from generator.generator import StaticSequenceGenerator
from randomness_test.basic_tests import frequency_test, poker_test, runs_test
from utils.unit_tools import nicer_time


def run_all(generator, continuous=False, print_log=False):
    """
    Run all FIPS 140-1 tests.
    """

    n = 20000

    # if we want all the tests to be applied to the *same* bit sequence,
    # we need to pre-compute it and create a static generator
    if not continuous:
        ts = time()
        sequence = generator.random_bytes((n + 1024) // 8)  # 1024 extra bits
        generator = StaticSequenceGenerator(seq=sequence)
        if print_log:
            print("(Sequence pre-computed in", nicer_time(time() - ts) + ')', flush=True)

    if not continuous:
        # print("(Rewinding sequence before next test)")
        generator.rewind()  # rewind

    misc = {}
    frequency_test(generator, n, misc=misc)
    t1 = 9654 < misc['n1'] < 10346

    if not continuous:
        # print("(Rewinding sequence before next test)")
        generator.rewind()  # rewind

    x3 = poker_test(generator, n, m=4)
    t2 = 1.03 < x3 < 57.4

    if not continuous:
        # print("(Rewinding sequence before next test)")
        generator.rewind()  # rewind

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
