# Random Bit/Number Generators and Randomness Statistical Tests

This project serves as a basis for the practical part of my bachelor thesis at the Vienna University of Technology. Currently, it is still a work in progress.

This project implements various pseudorandom and random number generators (RNGs). In most cases, I attempted to closely follow the given description, where available. In other cases, I implemented my own solutions, e.g. for the true random number generators (TRNGs) that utilize some of the sensors available on the computer (camera, audio input).

*Please note that the code here is for research purposes. Both good and bad RNGs are included here. I make no guarantees about the quality. Be careful before using any of the generators for scientific or cryptographic purposes!*

## Generators

Generators are in the module `generator`. All generators implement the default Python's `random.Random` class. Additionally, they also provide (at least) methods `random_bit()` and `random_byte()`. See abstract classes in `generator.generator` for signatures of various types of generators (in particular `BitGenerator` and `NumberGenerator`).

## Randomness Statistical Tests

Statistical tests for (pseudo)random number generators are in the module `randomness_test`. These can be used to test whether a given generator (or a sequence) can be considered "random".

## Implementation testing

Unittests of the generators and statistical tests are provided in the `test` module.

## Helper scripts

In the root directory, scripts `rng.py` and `run_all_tests.py` can be executed in shell to generate random numbers/bits and to test the generators. See their usage for details.

## Required Python and libraries

This project requires Python version 3. It has been developed and tested with version 3.5.

Additionally, following libraries are required:

* [NumPy](http://www.numpy.org/) and [SciPy](https://www.scipy.org/) for numerical calculations and analysis
* [pyDes](https://pypi.python.org/pypi/pyDes/) for 3DES encryption
* [Pygame](https://www.pygame.org/) for camera access for TRNG

On linux systems, you can install easily with pip:

    pip install numpy scipy pydes pygame

For other systems, refer to the links.

