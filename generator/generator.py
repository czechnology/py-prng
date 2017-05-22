import abc
from os.path import isfile
from random import Random

from utils.bit_tools import bits_to_byte
from utils.math_tools import is_power2, lcm


class RandomGenerator(Random, metaclass=abc.ABCMeta):
    def __init__(self, x):
        super().__init__(x)
        self.byte_buffer = []  # FIFO queue of bytes
        self.byte_buffer_pointer = 0
        self.bit_buffer = []  # FIFO queue of bits
        self.bit_buffer_pointer = 8

    def seed(self, a=None, version=2):
        self.byte_buffer = []
        self.byte_buffer_pointer = 0
        self.bit_buffer = []
        self.bit_buffer_pointer = 8

    @abc.abstractmethod
    def info(self):
        raise NotImplementedError('Generators must override method info() to use this base class')

    @abc.abstractmethod
    def random_bytes(self):
        raise NotImplementedError('Generators must define random_bytes to use this base class')

    def random_byte(self):
        if self.byte_buffer_pointer >= len(self.byte_buffer):  # if empty
            self.byte_buffer = self.random_bytes()
            self.byte_buffer_pointer = 0

        byte = self.byte_buffer[self.byte_buffer_pointer]
        self.byte_buffer_pointer += 1
        return byte

    def random_bit(self):
        """Return a random bit from the generator. Note that the default implementation of this
        method takes the random bit out from an internal bit buffer. Generators that produce bytes
        or numbers (rather than individual bytes) first fill up this buffer before a bit can be
        taken out of it. If you mix the use of random_byte() or random_bytes() methods with
        random_bit(), the produced sequence might not be as expected (although the randomness
        qualities should not be affected).
        """
        if self.bit_buffer_pointer >= 8:  # if empty
            self.bit_buffer = self.random_byte()
            self.bit_buffer_pointer = 0

        bit = (self.bit_buffer >> (7 - self.bit_buffer_pointer)) & 1
        self.bit_buffer_pointer += 1
        return bit

    def random(self):
        z = 0
        l = 8  # bytes

        for i in range(l):
            z = (z << 8) | self.random_byte()

        return z / (2 ** 64)

    def __str__(self) -> str:
        info = self.info()
        return info[0] + ' (' + '; '.join(info[1:]) + ')'


class BitGenerator(RandomGenerator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def random_bit(self):
        raise NotImplementedError('Generators must define random_bit to use this base class')

    def random_bytes(self):
        return [bits_to_byte([self.random_bit() for _ in range(8)])]

    def seed(self, a=None, version=2):
        super().seed(a, version)


class NumberGenerator(RandomGenerator, metaclass=abc.ABCMeta):
    def __init__(self, x=None):
        super().__init__(x)

        # the range bit bit generation needs to be limited to the next lower or equal power of two,
        # so that the the bit values share equal probability over all values (generated values over
        # this range must be discarded)

        if is_power2(self.max_value() + 1):  # if max_value = 2**e - 1
            self.max_bits = self.max_value().bit_length()
        else:
            self.max_bits = self.max_value().bit_length() - 1

        self.max_bits_value = (2 ** self.max_bits) - 1
        bits_extra = self.max_bits % 8
        self.rounds = lcm(bits_extra, 8) // bits_extra if bits_extra > 0 else 1

    @abc.abstractmethod
    def random_number(self):
        raise NotImplementedError('generators must define random_number to use this base class')

    @abc.abstractmethod
    def max_value(self):
        raise NotImplementedError('generators must define max_value to use this base class')

    def random(self):
        n = self.random_number() / (self.max_value() + 1)
        return n

    def random_bytes(self):
        max_bits, max_bits_value = self.max_bits, self.max_bits_value

        generated_bits = 0
        for i in range(self.rounds):
            number = self.random_number()
            while number > max_bits_value:
                number = self.random_number()

            generated_bits = (generated_bits << max_bits) | number

        return generated_bits.to_bytes(self.rounds * max_bits // 8, byteorder='big')


class StaticSequenceGenerator(RandomGenerator):
    NAME = "Generator repeating a given static sequence"

    def info(self):
        return [self.NAME,
                "sequence(%d): %s%s"
                % (self.n, str(self.sequence[:8]), "(first 8 bytes)" if self.n > 8 else ""),
                "seed (position): " + str(self.pos)]

    def __init__(self, seq, pos0=0):

        # check the sequence if it's really (probably) byte array and not mistakenly bit array
        if all(map(lambda b: b in [0, 1], seq)):
            raise ValueError('Error: The sequence is very likely to be a bit array instead of a'
                             'byte array')

        self.seq = seq
        self.n = len(seq)
        self.pos = None

        super().__init__(pos0)

    def seed(self, a=None, version=2):
        super().seed(a, version)
        self.pos = 0

    def rewind(self):
        self.seed(0)  # set position to the beginning of the sequence

    def random_bytes(self):
        byte = self.seq[self.pos]
        self.pos = (self.pos + 1) % self.n
        return [byte]

    def random(self):
        z = 0
        l = 8  # bytes

        for i in range(l):
            z = (z << 8) | self.random_byte()

        return z / (2 ** 64)

    def getstate(self):
        return self.VERSION, self.seq, self.pos

    def setstate(self, state):
        self.VERSION = state[0]
        self.seq = state[1]
        self.pos = state[2]


class StaticFileGenerator(RandomGenerator):
    NAME = "Generator repeating sequence from a given file"

    def info(self):
        return [self.NAME,
                "file: %s" % self.file,
                "seed (position): " + str(self.pos)]

    def __init__(self, file, pos0=0):
        if not isfile(file):
            raise ValueError("Specified file " + file + "doesn't exist")

        self.file = file
        self.file_handle = None

        super().__init__(pos0)

    # allow use of `with ... as` statement (see PEP 343)
    def __enter__(self):
        self.file_handle = open(self.file, 'rb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __del__(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def seed(self, a=None, version=2):
        super().seed(a, version)
        if self.file_handle:
            self.file_handle.seek(a)

    def random_bytes(self):
        bytes_ = self.file_handle.read(16)
        if not bytes_:
            print("Rewinding file")
            self.file_handle.seek(0)
            bytes_ = self.file_handle.read(16)

        return bytes_

    def random(self):
        z = 0
        l = 8  # bytes

        for i in range(l):
            z = (z << 8) | self.random_byte()

        return z / (2 ** 64)

    def getstate(self):
        return self.VERSION, self.file, self.file_handle.tell()

    def setstate(self, state):
        self.VERSION = state[0]
        self.file = state[1]
        self.file_handle = open(state[1], 'rb')
        self.file_handle.seek(state[2])
