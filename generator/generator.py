import abc
from os.path import isfile
from random import Random

from utils.bit_tools import bits_to_byte
from utils.math_tools import is_power2, lcm


class RandomGenerator(Random, metaclass=abc.ABCMeta):
    """
    General superclass of all our implemented generators. Provides methods to generate random bits
    and bytes and also implements the standard Python's random.Random class for simple conversion
    to other numerical formats.
    """

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
    def state(self):
        raise NotImplementedError('Generators must override method state() to use this base class')

    @abc.abstractmethod
    def random_bytes(self, min_bytes=1):
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

    def random_bytes_gen(self, n_bytes):
        chunk_size = 1024  # bytes
        generated_count = 0
        while generated_count < n_bytes:
            generated_chunk = self.random_bytes(chunk_size)
            for byte in generated_chunk:
                if generated_count < n_bytes:
                    generated_count += 1
                    yield byte

    def random_bits_gen(self, n_bits):
        chunk_size = 1024  # bytes
        generated_count = 0
        while generated_count < n_bits:
            generated_chunk = self.random_bytes(chunk_size)
            for byte in generated_chunk:  # iterate over bytes in chunk
                for i in reversed(range(8)):  # iterate over bits in byte
                    if generated_count < n_bits:
                        generated_count += 1
                        yield (byte >> i) & 1

    def __str__(self) -> str:
        info = self.info()
        return info[0] + ' (' + '; '.join(info[1:]) + ')'


class BitGenerator(RandomGenerator, metaclass=abc.ABCMeta):
    """
    Random bit generators which produce only single bits or sequences of bits in one iteration.
    """

    @abc.abstractmethod
    def random_bit(self):
        raise NotImplementedError('Generators must define random_bit to use this base class')

    def random_bytes(self, min_bytes=1):
        generated_bytes = []
        while len(generated_bytes) < min_bytes:
            byte = bits_to_byte([self.random_bit() for _ in range(8)])
            generated_bytes.append(byte)
        return generated_bytes

    def seed(self, a=None, version=2):
        super().seed(a, version)


class NumberGenerator(RandomGenerator, metaclass=abc.ABCMeta):
    """
    Random number generators which produce an integer number between 0 and max_value() in each
    iteration.
    """

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

    def random_bytes(self, min_bytes=1):
        max_bits, max_bits_value = self.max_bits, self.max_bits_value

        generated_bytes = []
        while len(generated_bytes) < min_bytes:
            generated_bits = 0
            for i in range(self.rounds):
                number = self.random_number()
                while number > max_bits_value:
                    number = self.random_number()

                generated_bits = (generated_bits << max_bits) | number

            generated_bytes += generated_bits.to_bytes(self.rounds * max_bits // 8, byteorder='big')

        return generated_bytes


class StaticSequenceGenerator(RandomGenerator):
    """
    A dummy generator producing bytes from a given static byte sequence. Once the end of the
    sequence is reached, the generator will rewind and start producing bytes from the start of the
    sequence again. This class is used mostly for testing.
    """

    NAME = "Generator repeating a given static sequence"

    def info(self):
        return [self.NAME,
                "sequence(%d): %s%s"
                % (self.n, str(self.sequence[:8]), "(first 8 bytes)" if self.n > 8 else ""),
                "seed (position): " + str(self.state())]

    def state(self):
        return self.pos

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

    def random_bytes(self, min_bytes=1):
        generated_bytes = []
        while len(generated_bytes) < min_bytes:
            # number of bytes to take from sequence
            n_bytes = min(min_bytes - len(generated_bytes), self.n - self.pos)
            generated_bytes += self.seq[self.pos:self.pos + n_bytes]
            self.pos = (self.pos + n_bytes) % self.n

        return generated_bytes

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
    """
    A dummy generator producing bytes from a given static binary file. Once the end of the file
    is reached, the generator will rewind and start producing bytes from the start of the file
    again. This class is used mostly for testing.
    """

    NAME = "Generator repeating sequence from a given file"

    def info(self):
        return [self.NAME,
                "file: %s" % self.file,
                "seed (position): " + str(self.state())]

    def state(self):
        return self.file_handle.tell() if self.file_handle else None

    def __init__(self, file, pos0=0):
        if not isfile(file):
            raise ValueError("Specified file " + file + "doesn't exist")

        self.file = file
        self.file_handle = None

        super().__init__(pos0)
        self.__enter__()

    # allow use of `with ... as` statement (see PEP 343)
    def __enter__(self):
        if not self.file_handle:
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

    def random_bytes(self, min_bytes=1):
        generated_bytes = []
        while len(generated_bytes) < min_bytes:
            # number of bytes to take from file
            n_bytes = min(min_bytes - len(generated_bytes), 1024)
            read_bytes = self.file_handle.read(n_bytes)
            if read_bytes:
                generated_bytes += read_bytes
            else:
                print("Rewinding file")
                self.file_handle.seek(0)

        return generated_bytes

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
