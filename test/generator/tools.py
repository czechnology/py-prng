def read_sequence(file):
    with open(file) as f:
        sequence = f.readlines()

    sequence = filter(lambda l: l[:1] != "#", sequence)
    sequence = map(lambda l: int(l.strip()), sequence)

    return list(sequence)


def number_sequences_to_compare(generator, file):
    expected_sequence = read_sequence(file)
    n = len(expected_sequence)
    generated_sequence = [generator.random_number() for _ in range(n)]
    return generated_sequence, expected_sequence
