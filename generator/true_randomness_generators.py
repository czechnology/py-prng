from hashlib import sha256
from struct import pack
from time import time

import pygame
import pygame.camera
import pygame.camera
# from pygame.locals import *
import pygame.image

from generator.generator import RandomGenerator
from utils.bit_tools import eliminate_bias

"""
"""


class CameraNoiseGenerator(RandomGenerator):
    """Implementation of a true random number generator. Instead of using purely algorithmic
    techniques based on the internal state like pseudorandom number generators, this generator
    uses input from a (covered) camera and uses the camera chip noise to collect randomness.
    To remove bias from the data, various techniques can be used. By default, a suitable one-way
    function (SHA256) is used.
    """

    NAME = 'Camera noise true randomness generator'

    def info(self):
        return [self.NAME,
                "Collecting data from camera"]

    def __init__(self):
        pygame.camera.init()
        available_cameras = pygame.camera.list_cameras()
        if not available_cameras:
            raise Exception("No camera available")
        self.cam = pygame.camera.Camera(available_cameras[0])
        self.last_raw = None
        self.last_time = None

        super().__init__(None)

    def seed(self, a=None, version=2):
        """Initialize the internal state of the generator."""
        if a:
            raise Exception("True randomness generator doesn't support seeding")

    def random_bytes(self, approach=3):

        # TODO evaluate more
        if approach == 0:
            raw = self._get_image_raw()
            d = eliminate_bias(raw)  # eliminate bias from data
            print("De-skewing: from", len(raw), "bytes to", len(d))
        elif approach == 20:
            raw1 = self._get_image_raw()
            raw2 = self._get_image_raw(True)
            d1 = eliminate_bias(raw1)  # eliminate bias from data
            d2 = eliminate_bias(raw2)  # eliminate bias from data
            d = list(map(lambda x: x[0] ^ x[1], zip(d1, d2)))
            print("De-skewing: from", 2 * len(raw1), "bytes to", len(d))
        elif approach == 1:
            raw = self._get_image_raw()
            m = sha256()
            m.update(raw)
            d = m.digest()
        elif approach == 2:
            raw = self._get_image_raw(ensure_fresh=True)  # each subsequent image must be different
            m = sha256()
            m.update(raw)
            d = m.digest()
        elif approach == 3:
            raw = self._get_image_raw()
            t = time()
            while t == self.last_time:
                t = time()
            self.last_time = t
            m = sha256()
            m.update(raw)
            m.update(pack("d", t))  # mix in time
            d = m.digest()
        else:
            raise Exception()

        return d

    def _get_image_raw(self, ensure_fresh=False):
        self.cam.start()
        raw = self.cam.get_image().get_buffer().raw
        while ensure_fresh and raw == self.last_raw:
            raw = self.cam.get_image().get_buffer().raw
        self.last_raw = raw
        self.cam.stop()
        return raw
