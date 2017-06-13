from math import ceil


def least_significant_bit(number, count=1):
    mask = (1 << count) - 1
    return int(number & mask)


def byte_xor(ba1, ba2):
    """Perform an XOR operation on two byte arrays"""
    return [b1 ^ b2 for (b1, b2) in zip(ba1, ba2)]


def left_rotate(n, b, bits=32):
    return least_significant_bit((n << b) | (n >> (bits - b)), bits)


def split_chunks(n, bits=32, pad=None):
    length = int(ceil(n.bit_length() / bits))
    chunks = [least_significant_bit(n >> bits * (length - i - 1), bits) for i in range(length)]

    if pad:
        # pad (if number had too many leading 0's)
        chunks = [0 for _ in range(abs(int(pad)) - len(chunks))] + chunks

    return chunks


def concat_chunks(chunks, bits=32):
    number = 0
    for ch in chunks:
        number = (number << bits) | ch
    return number


def bits_to_byte(bits):
    if len(bits) != 8:
        raise ValueError("Exactly 8 bits required")
    byte = 0
    for b in bits:
        if b & 1 != b:
            raise ValueError("The list of bits must consist only of 0's and 1's (was %s)"
                             % str(bits))
        byte = (byte << 1) | b
    return int(byte)


def byte_to_bits(byte):
    if byte != (byte & 0xff):
        raise ValueError("Byte value must be in range 0-255")
    return [(byte & 0x80) >> 7, (byte & 0x40) >> 6, (byte & 0x20) >> 5, (byte & 0x10) >> 4,
            (byte & 0x08) >> 3, (byte & 0x04) >> 2, (byte & 0x02) >> 1, (byte & 0x01)]


def eliminate_bias(bytes_):
    bytes_unbiased = []
    bits = []
    for bb in bytes_:
        for i in range(7, 0, -2):
            b1 = (bb >> i) & 1
            b2 = (bb >> i - 1) & 1
            if b1 != b2:
                bits.append(b1)
                if len(bits) >= 8:
                    bytes_unbiased.append(bits_to_byte(bits))
                    bits = []

    # superfluous bits get dropped, only complete bytes are returned

    return bytes_unbiased
