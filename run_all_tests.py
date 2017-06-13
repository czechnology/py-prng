#!/usr/bin/env python3

import re
import os
from argparse import ArgumentParser
from collections import OrderedDict
from sys import exit, stderr
from time import process_time as time

from generators import available_generators, create, reseed
from randomness_test import nist_sp_800_22_tests
from randomness_test import ent_tests as ent
from utils.unit_tools import nicer_time


def main():
    try:
        generator_ids, n_bits, seed, rounds, tex_path = parse_arguments()
    except ValueError as e:
        print(e, file=stderr)
        exit(1)
        return  # to remove inspection warnings

    print("RUN STAT TESTS\nGenerators: %s\nRounds: %s; Bits: %s; Seed: %s; TeX Path: %s\n%s"
          % (', '.join(generator_ids), "{:,}".format(rounds), "{:,}".format(n_bits), str(seed),
             tex_path, '=' * 80))

    run_all(generator_ids, n_bits, seed, rounds, tex_path)


def parse_arguments():
    parser = ArgumentParser(description='(Pseudo)Random Number Generators')
    parser.add_argument('-g', '--generator',
                        help='The ID of generator to be tested instead of file.')
    parser.add_argument('-f', '--file',
                        help='File with sequence to be tested instead of generator.')
    parser.add_argument('-n', '--bits', help='Total number of tested bits.', default='1M')
    parser.add_argument('-s', '--seed', help='Initial seed.')
    parser.add_argument('-r', '--rounds', help='Number of rounds to run each test.', default=1)
    parser.add_argument('-o', '--output', help='Output directory to write the results.')
    args = parser.parse_args()

    if args.seed:
        seed = int(args.seed)
    else:
        seed = None

    if (args.generator and args.file) or (not args.generator and not args.file):
        raise ValueError("Either generator or file should be specified")

    if args.file:
        if not os.path.isfile(args.file):
            raise ValueError("Source file " + args.file + " does not exist")
        generator_ids = ['file:' + args.file]
        seed = 0
    else:
        if args.generator == 'ALL':
            generator_ids = available_generators
        elif args.generator in available_generators:
            generator_ids = [args.generator]
        else:
            raise ValueError("Undefined generator. Available generators are:\n" +
                             ', '.join(available_generators))

    if args.bits.lower().endswith(('k', 'm', 'g')):
        n_bits = int(args.bits[:-1]) * (10 ** {'k': 3, 'm': 6, 'g': 9}[args.bits.lower()[-1:]])
    else:
        n_bits = int(args.bits)

    rounds = int(args.rounds)

    output_path = args.output
    if output_path and not os.path.isdir(output_path):
        raise ValueError("Specified TeX path " + output_path + " does not exist.")

    return generator_ids, n_bits, seed, rounds, output_path


def run_all(generator_ids, n_bits, seed=None, rounds=1, output_path=None):
    for gen_id in generator_ids:
        generator = create(gen_id, seed)
        # print('\n- '.join(generator.info()))
        print("Testing of generator", generator.NAME.upper())

        run_ent_tests(gen_id, generator, seed, n_bits, rounds, output_path)

        generator = create(gen_id, seed)  # recreate the generator
        run_nist_tests(gen_id, generator, seed, n_bits, rounds, output_path)

        # TODO: FIPS


def run_ent_tests(generator_id, generator, seed, n_bits, rounds, output_path):
    filename_pattern = 'gen-%s-ent-tests-%dr'
    filename = filename_pattern % (re.sub(r"[^A-Za-z0-9._-]+", '_', generator_id), rounds)

    # save the initial info about the generator (e.g. the seed)
    generator_info = generator.info()

    time_start = time()
    all_results = OrderedDict(
        [('state', []), ('entropy', []), ('chi_sq', []), ('mean', []), ('monte_pi', []), ('scc', [])])
    for i in range(rounds):
        print('-' * 80, "\nRUNNING ENT TESTS, round", i + 1)
        if seed is None:
            cur_seed = reseed(generator)
            print("New seed:", cur_seed)
        print('\n- '.join(generator.info()))
        entropy, chi_sq, mean, monte_pi, scc = ent.run_all(generator, n_bits, print_log=True)

        all_results['state'].append(generator.state())
        all_results['entropy'].append(entropy)
        all_results['chi_sq'].append(chi_sq)
        all_results['mean'].append(mean)
        all_results['monte_pi'].append(monte_pi)
        all_results['scc'].append(scc)

        if output_path:
            tsv = '# ENT statistical test results\n'
            tsv += '# !!! INTERMEDIATE DATA AFTER %d ROUNDS\n' % (i + 1)
            tsv += '# %s\n' % '\n# '.join(generator_info)
            tsv += generate_tsv_results(all_results)
            tsv += '\n# Final generator state: %s\n' % str(generator.getstate())
            tsv += '# Final generator info: %s\n' % ', '.join(generator_info)
            file = save_to_file(
                output_path, '.intermediary-'+filename, 'tsv', tsv)
            print("(intermediary TSV written to " + file + ')', flush=True)

    time_elapsed = time() - time_start

    print("*** Finished", rounds, "rounds of NIST tests for", "{:,}".format(n_bits),
          "bits each, in", nicer_time(time_elapsed), "seconds", flush=True)

    # add info
    info = generator_info + [
        '',  # empty line
        'ENT statistical test results',
        '%d rounds of %s bits long samples' % (rounds, "{:,}".format(n_bits)),
        'Time elapsed for generation and testing: ' + nicer_time(time_elapsed)]

    # generate TSV data tables
    tsv = '# ' + '\n# '.join(info) + '\n#\n'
    tsv += generate_tsv_results(all_results)
    tsv += '\n# Final generator state: %s\n' % str(generator.getstate())
    tsv += '# Final generator info: %s\n' % ', '.join(generator.info())
    if output_path:
        file = save_to_file(output_path, filename, 'tsv', tsv)
        print("TSV written to " + file, flush=True)
    else:
        print(tsv, flush=True)


def run_nist_tests(generator_id, generator, seed, n_bits, rounds, output_path):
    filename_pattern = 'gen-%s-nist-tests-%dr'
    filename = filename_pattern % (re.sub(r"[^A-Za-z0-9._-]+", '_', generator_id), rounds)

    # save the initial info about the generator (e.g. the seed)
    generator_info = generator.info()

    time_start = time()
    all_p_values = OrderedDict([('state', [])])
    for i in range(rounds):
        print('-' * 80, "\nRUNNING NIST TESTS, round", i + 1)
        if seed is None:
            cur_seed = reseed(generator)
            print("New seed:", cur_seed)
        print('\n- '.join(generator.info()))
        p_values = nist_sp_800_22_tests.run_all(generator, n_bits, print_log=True)

        all_p_values['state'].append(generator.state())
        for pval, test_id, test_name in p_values:
            if test_id not in all_p_values:
                all_p_values[test_id] = []
            all_p_values[test_id].append(pval)

        if output_path:
            tsv = '# NIST SP 800-22 statistical test results\n'
            tsv += '# !!! INTERMEDIATE DATA AFTER %d ROUNDS\n' % (i + 1)
            tsv += '# %s\n' % '\n# '.join(generator_info)
            tsv += generate_tsv_results(all_p_values)
            tsv += '\n# Final generator state: %s\n' % str(generator.getstate())
            tsv += '# Final generator info: %s\n' % ', '.join(generator_info)
            file = save_to_file(
                output_path, '.intermediary-'+filename, 'tsv', tsv)
            print("(intermediary TSV written to " + file + ')', flush=True)

    time_elapsed = time() - time_start

    print("*** Finished", rounds, "rounds of NIST tests for", "{:,}".format(n_bits),
          "bits each, in", nicer_time(time_elapsed), "seconds", flush=True)

    # add info
    info = generator_info + [
        '',  # empty line
        'NIST SP 800-22 statistical test results',
        '%d rounds of %s bits long samples' % (rounds, "{:,}".format(n_bits)),
        'Time elapsed for generation and testing: ' + nicer_time(time_elapsed)]

    # generate TSV data tables
    tsv = '# ' + '\n# '.join(info) + '\n#\n'
    tsv += generate_tsv_results(all_p_values)
    tsv += '\n# Final generator state: %s\n' % str(generator.getstate())
    tsv += '# Final generator info: %s\n' % ', '.join(generator.info())
    if output_path:
        file = save_to_file(output_path, filename, 'tsv', tsv)
        print("TSV written to " + file, flush=True)
    else:
        print(tsv, flush=True)


def generate_tsv_results(data):
    tsv = ''

    for test_name, test_values in data.items():
        if not test_values:
            continue

        if type(test_values[0]) in (list, tuple):
            test_values = [','.join(map(str, vv)) for vv in test_values]

        line = [test_name] + test_values

        tsv += '\t'.join(map(str, line)) + '\n'

    return tsv


def save_to_file(path, basename, ext, data):
    if not ext.startswith('.'):
        ext = '.' + ext
    if not os.path.isdir(path):
        raise ValueError("Path " + path + " is not a valid directory")
    basename = basename.replace('_', '-')  # underscores are unsafe in TeX documents
    filename = get_available_filename(path + '/' + basename, ext)
    with open(filename, 'w') as f:
        f.write(data)
    return filename


def get_available_filename(file, ext):
    if not os.path.exists(file + ext):
        return file + ext
    patt = '%s(%d)%s'
    i = 2
    filename = patt % (file, i, ext)
    while os.path.exists(filename):
        i += 1
        filename = patt % (file, i, ext)
    return filename


if __name__ == '__main__':
    main()
