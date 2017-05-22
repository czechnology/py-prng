# from itertools import grouper
from random import SystemRandom

from generator import linear_congruential_generators as lcg
from generator.cryptographically_secure_generators import RsaGenerator, MicaliSchnorrGenerator, \
    BlumBlumShubGenerator
from generator.generalised_feedback_shift_registers import MersenneTwister32, MersenneTwister64
from generator.permuted_congruential_generators import PermutedCongruentialGenerator

available_generators = (
    # Cryptographically secure generators
    'rsa', 'mic_sch', 'bbs',
    # Generalised feedback shift registers
    'mt32', 'mt64',
    # Linear congruential generators
    'lcg_java', 'lcg_randu', 'lcg_simple',
    # Permuted congruential generators
    'pcg'
)


def create(generator_id, seed=None):
    if not generator_id:
        raise ValueError("No generator selected")

    if not seed:
        seed = SystemRandom().getrandbits(128)

    # Cryptographically secure generators
    if generator_id == 'rsa':
        p, q, e = 61, 53, 17
        generator = RsaGenerator((p, q, e), seed=seed % (p * q))
    elif generator_id == 'mic_sch':
        p, q, e = 61, 53, 17
        generator = MicaliSchnorrGenerator((p, q, e), seed=seed % (p * q))
    elif generator_id == 'bbs':
        p, q = 7027, 611207
        generator = BlumBlumShubGenerator((p, q), seed=seed % (p * q))

    # Generalised feedback shift registers
    elif generator_id == 'mt32':
        generator = MersenneTwister32(seed=seed)
    elif generator_id == 'mt64':
        generator = MersenneTwister64(seed=seed)

    # Linear congruential generators
    elif generator_id == 'lcg_java':
        generator = lcg.JavaLinearCongruentialGenerator(seed=seed % (2 ** 48))
    elif generator_id == 'lcg_randu':
        generator = lcg.RanduLinearCongruentialGenerator(seed=seed % (2 ** 31))
    elif generator_id == 'lcg_simple':
        m, a, c = 10, 7, 7
        generator = lcg.LinearCongruentialGenerator(m=m, a=a, c=c, seed=seed % m)

    # Permuted congruential generators
    elif generator_id == 'pcg':
        generator = PermutedCongruentialGenerator(12345, seed=seed)

    # No (or unknown) generator selected
    else:
        raise ValueError("Invalid generator")

    return generator
