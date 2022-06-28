import numpy as np
from numpy.random import Generator, SeedSequence, PCG64
import hashlib


def get_rng(seed: str, verbose=False):
    """Generate a random number generator based on a seed string."""
    # Over python iteration the traditional hash was changed. So, here we fix it to md5
    hash = hashlib.md5(seed.encode("utf-8")).hexdigest()  # Convert string to a hash
    seed_int = int(hash, 16) % (10 ** 6)  # Convert hash to an fixed size integer
    if verbose:
        print("Seed to md5 hash:", seed, "->", hash, "->", seed_int)
    # Create instance of random number generator explicitly to ensure long time support
    # PCG64 -> https://www.pcg-random.org/
    # see https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    rng = Generator(PCG64(SeedSequence(seed_int)))
    return rng
