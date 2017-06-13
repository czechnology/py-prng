from math import log2
from math import pi
from time import process_time as time

from scipy.stats import chi2

from generator.generator import StaticSequenceGenerator
from utils.bit_tools import bits_to_byte
from utils.unit_tools import nicer_time

"""
A python port of ENT tests by John Walker.
Website and original C code: http://www.fourmilab.ch/random/

TODO: "pythonize" the code more.
"""


def run_all(generator, n, binary=True, continuous=False, print_log=False):
    # if we want all the tests to be applied to the *same* bit sequence,
    # we need to pre-compute it and create a static generator
    if not continuous:
        ts = time()
        sequence = generator.random_bytes((n // 8 if binary else n) + 128)  # 1024 extra bits
        generator = StaticSequenceGenerator(seq=sequence)
        if print_log:
            print("(Sequence pre-computed in", nicer_time(time() - ts) + ')', flush=True)

    prob = [0.0] * 256
    MONTEN = 6
    monte = []

    # Initialise for calculations
    ent = 0.0  # Clear entropy accumulator
    chisq = 0.0  # Clear Chi-Square
    datasum = 0.0  # Clear sum of bytes for arithmetic mean

    mcount = 0  # Clear Monte Carlo tries
    inmont = 0  # Clear Monte Carlo inside count

    sccfirst = True  # Mark first time for serial correlation
    scct1 = scct2 = scct3 = 0.0  # Clear serial correlation terms

    incirc = ((256.0 ** (MONTEN // 2)) - 1) ** 2

    ccount = [0] * 256

    totalc = 0

    c_gen = generator.random_bits_gen if binary else generator.random_bytes_gen
    bits = []
    for c in c_gen(n):
        ccount[c] += 1  # Update counter for this bin
        totalc += 1

        if binary:
            bits.append(c)
            if len(bits) == 8:
                byte = bits_to_byte(bits)
                bits = []
            else:
                byte = None
        else:
            byte = c

        # Update inside / outside circle counts for Monte Carlo computation of PI
        if byte is not None:
            monte.append(byte)  # Save character for Monte Carlo
            if len(monte) >= MONTEN:  # Calculate every MONTEN character
                mcount += 1
                montex = montey = 0
                for mj in range(MONTEN // 2):
                    montex = (montex * 256.0) + monte[mj]
                    montey = (montey * 256.0) + monte[(MONTEN // 2) + mj]
                if (montex * montex + montey * montey) <= incirc:
                    inmont += 1

                # print(monte, '\t', montex, montey, inmont)
                # if mcount >= 10:
                #     sys.exit()

                # clear the buffer
                monte = []

        # Update calculation of serial correlation coefficient
        sccun = c
        if sccfirst:
            sccfirst = False
            sccu0 = sccun
        else:
            scct1 = scct1 + scclast * sccun

        scct2 = scct2 + sccun
        scct3 = scct3 + (sccun * sccun)
        scclast = sccun

    # Complete calculation of serial correlation coefficient
    scct1 = scct1 + scclast * sccu0
    scct2 = scct2 * scct2
    scc = totalc * scct3 - scct2
    if scc == 0.0:
        scc = -100000
    else:
        scc = (totalc * scct1 - scct2) / scc

    # Scan bins and calculate probability for each bin and Chi-Square distribution. The probability
    # will be reused in the entropy calculation below.  While we're at it, we sum of all the data
    # which will be used to compute the mean.
    cexp = totalc / (2 if binary else 256)  # Expected count per bin
    for i in range(2 if binary else 256):
        a = ccount[i] - cexp
        prob[i] = ccount[i] / totalc
        chisq += (a * a) / cexp
        datasum += i * ccount[i]

    # Calculate entropy
    for i in range(2 if binary else 256):
        if prob[i] > 0:
            ent += prob[i] * log2(1 / prob[i])

    # Calculate Monte Carlo value for PI from percentage of hits within the circle
    montepi = 4.0 * (inmont / mcount)

    mean = datasum / totalc

    samp = "bit" if binary else "byte"
    samp_bits = 1 if binary else 8
    chip = chi2.sf(chisq, df=(1 if binary else 255))

    if print_log:
        text_output = "Entropy = %f bits per %s.\n" % (ent, "bit" if binary else "byte")
        text_output += "Optimum compression would reduce the size of this %d %s file by %d percent.\n" \
                       % (n, samp, ((100 * (samp_bits - ent)) / samp_bits))
        text_output += "Chi square distribution for %d samples is %1.2f, and randomly would exceed " \
                       "this value " % (totalc, chisq)
        if chip < 0.0001:
            text_output += " less than 0.01 percent of the times.\n"
        elif chip > 0.9999:
            text_output += " more than than 99.99 percent of the times.\n"
        else:
            text_output += "%1.2f percent of the times.\n" % (chip * 100)

        text_output += "Arithmetic mean value of data %ss is %1.4f (%.1f = random).\n" \
                       % (samp, mean, .5 if binary else 127.5)

        text_output += "Monte Carlo value for Pi is %1.9f (error %1.2f percent).\n" \
                       % (montepi, 100.0 * (abs(pi - montepi) / pi))

        text_output += "Serial correlation coefficient is " + \
                       ("%1.6f (totally uncorrelated = 0.0).\n" % scc if scc >= -99999
                        else "undefined (all values equal!).\n")
        print(text_output)

    # Return results through arguments
    return ent, chisq, mean, montepi, scc
