#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser
from os import linesep
from random import SystemRandom
from signal import signal, SIGPIPE, SIG_DFL
from time import process_time as time

from generators import available_generators, create
from utils.console_tools import query_yes_no_abort


def main():
    parser = ArgumentParser(description='(Pseudo)Random Number Generators')
    parser.add_argument('-g', '--generator', help='Generator selection.')
    parser.add_argument('-s', '--seed', help='Initial seed.')
    parser.add_argument('-c', '--count', help='Total number of generated numbers.', default='10')
    parser.add_argument('-b', '--binary', help='Binary mode.', action='store_true')
    parser.add_argument('-t', '--timer', help='Timer mode (no output)', action='store_true')
    parser.add_argument('-f', '--file', help='File for output')
    args = parser.parse_args()

    args.count = args.count.lower()
    if args.count == 'inf':
        count = 2 ** 50
    elif args.count.endswith('k'):
        count = 1000 * int(args.count[:-1])
    elif args.count.endswith('m'):
        count = 1000000 * int(args.count[:-1])
    elif args.count.endswith('g'):
        count = 1000000000 * int(args.count[:-1])
    else:
        count = int(args.count)

    if args.seed:
        seed = int(args.seed)
    else:
        seed = SystemRandom().getrandbits(128)

    if args.generator not in available_generators:
        print("Invalid generator. Available values are", available_generators)
        parser.print_usage()
        sys.exit(1)

    generator = create(args.generator, seed)

    out = None
    if not args.timer:
        if args.file:
            if os.path.isfile(args.file) and not query_yes_no_abort(
                                    "Specified file " + args.file + " exists. Overwrite?"):
                print('Aborting')
                sys.exit(0)
            out = os.open(args.file, flags=os.O_WRONLY | os.O_BINARY | os.O_CREAT)
        else:
            out = sys.stdout.fileno()

    time_start = time()
    if args.binary:
        # def gen_byte
        if args.timer:
            list(map(lambda j: (generator.random_byte()), range(count)))
        else:
            remaining = count
            while remaining > 0:
                generated_bytes = generator.random_bytes()
                os.write(out, bytes(generated_bytes))
                remaining -= len(generated_bytes)

    else:  # not args.binary
        if hasattr(generator, 'random_number'):
            random_numbers = map(lambda j: (generator.random_number()), range(count))
        else:
            random_numbers = map(lambda j: (generator.randrange(0,2<<32)), range(count))

        if args.timer:
            list(random_numbers)
        else:
            os.write(out, bytes(linesep.join(map(str, random_numbers)), 'ascii'))

    if args.file or args.timer:
        time_elapsed = time() - time_start
        print("Finished in %.3f seconds" % time_elapsed, end='')
        if args.binary and time_elapsed > 0:
            print(',', 8 * count / time_elapsed / 1024 / 1024, "Mbps", end='')
        print()


def print_progress(perc):
    print(str(perc) + '% ', end='', flush=True)


if __name__ == '__main__':
    signal(SIGPIPE, SIG_DFL)
    main()
