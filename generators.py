from random import SystemRandom

from generator import cryptographically_secure_generators as csprng
from generator import generalised_feedback_shift_registers as gfsr
from generator import linear_congruential_generators as lcg
from generator import linear_feedback_shift_registers as lfsr
from generator import one_way_function_generators as owfg
from generator import permuted_congruential_generators as pcg
from generator import true_randomness_generators as trng
from generator.generator import StaticFileGenerator, StaticSequenceGenerator

available_generators = (
    # Cryptographically secure generators
    'rsa', 'mic_sch', 'bbs',
    # Generalised feedback shift registers
    'mt32', 'mt64',
    # Linear congruential generators
    'lcg_java', 'lcg_randu', 'lcg_simple',
    # Linear feedback shift register
    'lfsr_131',
    # One-way function generators
    'ansix917', 'fips186_pk_sha', 'fips186_pk_des', 'fips186_msg_sha', 'fips186_msg_des',
    # Permuted congruential generators
    'pcg',
    # True random number generators
    'cam', 'mic',
)


def create(generator_id, seed=None):
    if not generator_id:
        raise ValueError("No generator selected")

    if generator_id.startswith('file:'):
        print(generator_id[5:], seed)
        return StaticFileGenerator(generator_id[5:])

    if not seed:
        seed = SystemRandom().getrandbits(128)

    # Cryptographically secure generators
    if generator_id == 'rsa':
        p, q, e = 61, 53, 17
        return csprng.RsaGenerator((p, q, e), seed=seed % (p * q))
    elif generator_id == 'mic_sch':
        p, q, e = 61, 53, 17
        return csprng.MicaliSchnorrGenerator((p, q, e), seed=seed % (p * q))
    elif generator_id == 'bbs':
        p, q = 7027, 611207
        return csprng.BlumBlumShubGenerator((p, q), seed=seed % (p * q))

    # Generalised feedback shift registers
    elif generator_id == 'mt32':
        return gfsr.MersenneTwister32(seed=seed)
    elif generator_id == 'mt64':
        return gfsr.MersenneTwister64(seed=seed)

    # Linear congruential generators
    elif generator_id == 'lcg_java':
        return lcg.JavaLinearCongruentialGenerator(seed=seed % (2 ** 48))
    elif generator_id == 'lcg_randu':
        return lcg.RanduLinearCongruentialGenerator(seed=seed % (2 ** 31))
    elif generator_id == 'lcg_simple':
        m, a, c = 10, 7, 7
        return lcg.LinearCongruentialGenerator(m=m, a=a, c=c, seed=seed % m)

    # Linear feedback shift registers
    elif generator_id == 'lfsr_131':
        return lfsr.LinearFeedbackShiftRegister(131, taps=(131, 130, 84, 83),
                                                seed=seed % (2 ** 131))

    # One-way function generators:
    elif generator_id == 'ansix917':
        return owfg.AnsiX917Generator(seed=seed % (2 ** 64))
    elif generator_id == 'fips186_pk_sha':
        return owfg.Fips186GeneratorPk(g=owfg.Fips186Generator.OWF_SHA1, seed=seed % (2 ** 160))
    elif generator_id == 'fips186_pk_des':
        return owfg.Fips186GeneratorPk(g=owfg.Fips186Generator.OWF_DES, seed=seed % (2 ** 160))
    elif generator_id == 'fips186_msg_sha':
        return owfg.Fips186GeneratorMsg(g=owfg.Fips186Generator.OWF_SHA1, seed=seed % (2 ** 160))
    elif generator_id == 'fips186_msg_des':
        return owfg.Fips186GeneratorMsg(g=owfg.Fips186Generator.OWF_DES, seed=seed % (2 ** 160))

    # Permuted congruential generators
    elif generator_id == 'pcg':
        return pcg.PermutedCongruentialGenerator(12345, seed=seed)

    # # True random number generators
    elif generator_id == 'cam':
        return trng.CameraNoiseGenerator()
    elif generator_id == 'mic':
        return trng.MicrophoneNoiseGenerator()

    # No (or unknown) generator selected
    else:
        raise ValueError("Invalid generator")


def reseed(generator):
    if not generator:
        raise ValueError("No generator given")

    generator_type = type(generator)

    if generator_type in (StaticFileGenerator, StaticSequenceGenerator):
        return

    seed = SystemRandom().getrandbits(128)

    # Cryptographically secure generators
    if generator_type in (
            csprng.RsaGenerator, csprng.MicaliSchnorrGenerator, csprng.BlumBlumShubGenerator):
        seed = seed % generator.n

    # Generalised feedback shift registers
    elif generator_type is gfsr.MersenneTwister32:
        seed = seed % (2 ** 32)
    elif generator_type is gfsr.MersenneTwister64:
        seed = seed % (2 ** 64)

    # Linear congruential generators
    elif isinstance(generator, lcg.LinearCongruentialGenerator):
        seed = seed % generator.m

    # Linear feedback shift registers
    elif generator_type is lfsr.LinearFeedbackShiftRegister:
        seed = seed % (2 ** generator.length)

    # One-way function generators:
    elif generator_type is owfg.AnsiX917Generator:
        seed = seed % (2 ** 64)
    elif isinstance(generator, owfg.Fips186Generator):
        seed = seed % (2 ** 160)

    # Permuted congruential generators
    elif generator_type is pcg.PermutedCongruentialGenerator:
        seed = seed % (2 ** 64)

    # True random number generators
    elif generator_type in (trng.CameraNoiseGenerator,  trng.MicrophoneNoiseGenerator):
        pass

    # No (or unknown) generator selected
    else:
        raise ValueError("Invalid generator")

    generator.seed(seed)

    return seed
