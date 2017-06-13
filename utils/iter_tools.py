from itertools import zip_longest


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks.
    grouper('ABCDEFG', 3, 'x') == ['ABC','DEF','Gxx']
    If fillvalue is None, skip last block that isn't full:
    grouper('ABCDEFG', 3) == ['ABC','DEF']

    Inspired by https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    args = [iter(iterable)] * n

    if fillvalue is None:
        return zip(*args)
    else:
        return zip_longest(*args, fillvalue=fillvalue)
