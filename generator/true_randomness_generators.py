import abc
import operator
from functools import reduce
from hashlib import sha256

import pyaudio
import pygame
import pygame.camera
import pygame.camera
import pygame.image

from generator.generator import RandomGenerator
from utils.iter_tools import grouper

"""
An attempt to implement true random number generators utilizing only sensors available on most
consumer mobile devices - camera and microphone.
"""


class NoiseGenerator(RandomGenerator, metaclass=abc.ABCMeta):
    """
    A random number generator based on the data from a hardware sensor.
    """

    def state(self):
        return None

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator."""
        if a:
            raise Exception("True randomness generator doesn't support seeding")

    @staticmethod
    def _extract_randomness_sha(raw, chunk_size):
        extracted_bytes = []
        for chunk in grouper(raw, chunk_size):
            m = sha256()
            m.update(bytes(chunk))
            extracted_bytes.append(m.digest())

        extracted_bytes = reduce(operator.add, extracted_bytes)

        return extracted_bytes


class CameraNoiseGenerator(NoiseGenerator):
    """Implementation of a true random number generator. Instead of using purely algorithmic
    techniques based on the internal state like pseudorandom number generators, this generator
    uses input from a camera and uses the camera input and chip noise to collect randomness.
    To remove bias from the data, a suitable one-way function (SHA256) is used.
    """

    NAME = 'Camera noise true randomness generator'

    def info(self):
        return [self.NAME,
                "Collecting data from camera"]

    def __init__(self, chunk_size=239):
        pygame.camera.init()
        available_cameras = pygame.camera.list_cameras()
        if not available_cameras:
            raise Exception("No camera available")
        self.cam = pygame.camera.Camera(available_cameras[0])
        self.last_raw = None
        self.last_time = None
        self.chunk_size = chunk_size

        super().__init__(None)

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator."""
        if a:
            raise Exception("True randomness generator doesn't support seeding")

    def random_bytes(self, min_bytes=1):
        generated_bytes = []
        self.cam.start()
        while len(generated_bytes) < min_bytes:
            raw = self._get_image_raw(ensure_fresh=True)  # each subsequent image must be different
            chunk_size = self.chunk_size
            extracted_bytes = self._extract_randomness_sha(raw, chunk_size)
            generated_bytes.extend(extracted_bytes)
        self.cam.stop()

        return generated_bytes

    def _get_image_raw(self, ensure_fresh=False):
        raw = self.cam.get_image().get_buffer().raw
        while ensure_fresh and raw == self.last_raw:
            raw = self.cam.get_image().get_buffer().raw
        self.last_raw = raw
        return raw


class MicrophoneNoiseGenerator(NoiseGenerator):
    """Implementation of a true random number generator. Instead of using purely algorithmic
    techniques based on the internal state like pseudorandom number generators, this generator
    uses input from a microphone and uses the audio input to collect randomness.
    To remove bias from the data, a suitable one-way function (SHA256) is used.
    """

    NAME = 'Microphone Noise True Randomness Generator'
    SKIP = 10
    FORMAT = pyaudio.paInt32
    CHANNELS = 1
    RATE = 192000
    FPB = 2048

    def info(self):
        return [self.NAME,
                "Collecting data from microphone (audio input)"]

    def __init__(self, chunk_size=281):
        self.pa = pyaudio.PyAudio()
        self.chunk_size = chunk_size
        super().__init__(None)

    def __del__(self):
        self.pa.terminate()

    def find_input_devices(self):
        # sample rate discovery based on https://stackoverflow.com/a/11837434
        standard_sample_rates = [8000.0, 9600.0, 11025.0, 12000.0, 16000.0, 22050.0, 24000.0,
                                 32000.0, 44100.0, 48000.0, 88200.0, 96000.0, 192000.0]

        for i in range(self.pa.get_device_count()):
            dev_info = self.pa.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                supported_sample_rates = []
                for f in standard_sample_rates:
                    try:
                        if self.pa.is_format_supported(
                                f,
                                input_device=dev_info['index'],
                                input_channels=dev_info['maxInputChannels'],
                                input_format=pyaudio.paInt16):
                            supported_sample_rates.append(f)
                    except ValueError:
                        pass

                print("Device %d: %s, supported sample_rates: %s"
                      % (i, dev_info["name"], str(supported_sample_rates)))

    def random_bytes(self, min_bytes=1):
        def open_stream():
            stream = self.pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.FPB
            )
            # skip a few chunks in the beginning which often show signs of non-randomness
            for _ in range(self.SKIP):
                stream.read(self.FPB)
            return stream

        stream = open_stream()

        io_errors = skipped = 0
        generated_bytes = []
        while len(generated_bytes) < min_bytes and io_errors < 100 and skipped < 50:
            try:
                raw = stream.read(self.FPB)
            except IOError:
                # try reopening the stream
                stream = open_stream()
                io_errors += 1
                continue

            extracted_bytes = self._extract_randomness_sha(raw, self.chunk_size)
            generated_bytes.extend(extracted_bytes)

        try:
            stream.stop_stream()
            stream.close()
        except OSError:
            pass

        return generated_bytes
