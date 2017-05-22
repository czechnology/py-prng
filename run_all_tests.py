#!/usr/bin/env python3

import os
from argparse import ArgumentParser
from collections import OrderedDict
from functools import reduce
from random import SystemRandom
from sys import exit, stderr
from time import process_time as time

from generators import available_generators, create
from randomness_test import nist_sp_800_22_tests
from utils.stat_tools import boxplot_params
from utils.unit_tools import nicer_time


def main():
    try:
        generator_ids, n_bits, seed, rounds, tex_path = parse_arguments()
    except ValueError as e:
        print(e, file=stderr)
        exit(1)
        return  # to remove inspection warnings

    print("RUN STAT TESTS\nGenerators: %s\nRounds: %s; Bits: %s; Seed: %d; TeX Path: %s\n%s"
          % (', '.join(generator_ids), "{:,}".format(rounds), "{:,}".format(n_bits), seed,
             tex_path, '=' * 80))

    run_all(generator_ids, n_bits, seed, rounds, tex_path)


def parse_arguments():
    parser = ArgumentParser(description='(Pseudo)Random Number Generators')
    parser.add_argument('-g', '--generator', help='The generator ID.')
    parser.add_argument('-n', '--bits', help='Total number of tested numbers.', default='1M')
    parser.add_argument('-s', '--seed', help='Initial seed.')
    parser.add_argument('-r', '--rounds', help='Number of rounds to run each test.', default=1)
    parser.add_argument('-o', '--output', help='Output directory to write the results.')
    args = parser.parse_args()

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

    if args.seed:
        seed = int(args.seed)
    else:
        seed = SystemRandom().getrandbits(128)

    rounds = int(args.rounds)

    output_path = args.output
    if output_path and not os.path.isdir(output_path):
        raise ValueError("Specified TeX path " + output_path + " does not exist.")

    return generator_ids, n_bits, seed, rounds, output_path


def run_all(generator_ids, n_bits, seed=None, rounds=1, output_path=None):
    for gen_id in generator_ids:
        generator = create(gen_id, seed)
        print('\n- '.join(generator.info()))

        run_nist_tests(gen_id, generator, n_bits, rounds, output_path)

        # TODO
        # # FIPS
        # p_values
        # for i in range(rounds):
        #     p_values = nist_sp_800_22_tests.run_all(generator, n_bits, print_log=True)

        # rand = Random(12345)


def run_nist_tests(generator_id, generator, n_bits, rounds, output_path):

    # save the initial info about the generator (e.g. the seed)
    generator_info = generator.info()

    time_start = time()
    all_p_values = OrderedDict()
    for i in range(rounds):
        print('-' * 80, "\nRUNNING NIST TESTS, round", i + 1)
        p_values = nist_sp_800_22_tests.run_all(generator, n_bits, print_log=True)

        for pval, test_id, test_name in p_values:
            if test_id not in all_p_values:
                all_p_values[test_id] = []
            all_p_values[test_id].append(pval)

        if output_path:
            tsv = '# NIST SP 800-22 statistical test results\n'
            tsv += '# !!! INTERMEDIATE DATA AFTER %d ROUNDS\n' % (i + 1)
            tsv += '# %s\n' % '\n# '.join(generator_info)
            tsv += generate_tsv_pvalues(all_p_values)
            tsv += '\n# Final generator state: %s\n' % str(generator.getstate())
            tsv += '# Final generator info: %s\n' % ', '.join(generator_info)
            file = save_to_file(
                output_path, '.intermediary-gen-' + generator_id + '-nist-800-22-tests', 'tsv', tsv)
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
    tsv += generate_tsv_pvalues(all_p_values)
    tsv += '\n# Final generator state: %s\n' % str(generator.getstate())
    tsv += '# Final generator info: %s\n' % ', '.join(generator_info)
    if output_path:
        file = save_to_file(output_path, 'gen-' + generator_id + '-nist-800-22-tests', 'tsv', tsv)
        print("TSV written to " + file, flush=True)
    else:
        print(tsv, flush=True)

    # TODO
    # # generate TeX box plots
    # tex = '% ' + '\n% '.join(info) + '\n\n'
    # tex += generate_boxplot_pvalues(all_p_values, nist_sp_800_22_tests.all_tests)
    # if output_path:
    #     file = save_to_file(
    #         output_path, 'gen-' + generator_id + '-nist-800-22-tests-boxplots', 'tex', tex)
    #     print("TeX written to " + file, flush=True)
    # else:
    #     print(tex, flush=True)


def generate_tsv_pvalues(data):
    # tsv = 'test_name, mean, var, min, q1, med, q3, max, count, p_values'.replace(', ', '\t') +'\n'
    tsv = ''

    for test_name, test_values in data.items():
        if not test_values:
            continue

        if type(test_values[0]) in (list, tuple):
            # test_values = reduce(operator.add, map(list, test_values))
            test_values = [','.join(map(str, vv)) for vv in test_values]

        # values_array = np.array(test_values)
        # desc = sp.stats.describe(values_array)
        # q1, med, q3 = np.percentile(values_array, [25, 50, 75], interpolation='midpoint')

        line = [
            test_name,
            # desc.mean, desc.variance,
            # desc.minmax[0], q1, med, q3, desc.minmax[1],
            # desc.nobs
        ]
        line += test_values

        tsv += '\t'.join(map(str, line)) + '\n'

    return tsv


def generate_boxplot_pvalues(data, data_labels=(), standalone=True):
    tex_document_start_tpl = r"""
\documentclass{standalone}
\usepackage{tikz,pgfplots}
\usepgfplotslibrary{statistics}
\pgfplotsset{
    compat=1.14,
    every boxplot/.style={mark=o,solid,color=black,fill=white},
    boxplot/box extend=0.6
}
\begin{document}
    """.strip() + '\n'

    def tex_picture_start_tpl(tick_labels, count):
        tpl = r"""
\begin{tikzpicture}
\begin{axis}[
    ytick={%s},
    yticklabels={%s},
    y=1.5em, xmin=0, xmax=1, tick style={draw=none}, xmajorgrids=true, xlabel={P-value},
    ymin=0.4, ymax=%.1f
]
        """.strip() + '\n'

        return tpl % (
            ','.join(map(str, range(1, len(tick_labels) + 1))),
            ','.join(tick_labels),
            count + .6
        )

    tex_picture_end_tpl = r"""
\end{axis}
\end{tikzpicture}
    """.strip()

    tex_document_end_tpl = r"""
\end{document}
    """.strip()

    def tex_plot_tpl(i, name, values, lq, med, uq, lw, uw, out):
        tpl = r"""
\addplot[
    boxplot prepared={
        lower quartile=%.6f, median=%.6f, upper quartile=%.6f,
        lower whisker=%.6f, upper whisker=%.6f
    },
] coordinates { %s };
    """.strip() + '\n'

        tpl_small = r"""
\addplot[mark=*,color=red] coordinates { %s } ;
        """.strip() + '\n'

        tex = ('\n%% %s\n%% Calculated P-values: %s\n'
               % (name, ', '.join(str(round(v, 6)) for v in values))) + \
              (tpl % (lq, med, uq, lw, uw, ' '.join(("(%d,%.6f)" % (i, v)) for v in out)))

        too_small = set(filter(lambda p: p < 0.01, [bp['lw']] + bp['out']))
        if too_small:
            tex += tpl_small % (' '.join(("(%.6f,%d)" % (v, i)) for v in too_small))

        return tex

    labels = []
    tex_plots = ""
    # the plots are displayed on y=1,2,3,..., i.e. from bottom to top, so reverse the list
    for test_id, test_values in reversed(data.items()):
        if not test_values:
            continue
        if type(test_values[0]) in (list, tuple):
            test_values = reduce(lambda v1, v2: list(v1) + list(v2), test_values)

        label = data_labels.get(test_id, test_id)
        labels.append(label)
        bp = boxplot_params(test_values)
        tex_plots += tex_plot_tpl(len(labels), label, test_values,
                                  bp['lq'], bp['med'], bp['uq'], bp['lw'], bp['uw'], bp['out'])

    tex = tex_document_start_tpl + '\n' if standalone else ''
    tex += tex_picture_start_tpl(labels, len(labels))
    tex += tex_plots + '\n'
    tex += tex_picture_end_tpl + '\n'
    tex += tex_document_end_tpl + '\n' if standalone else ''

    return tex


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
