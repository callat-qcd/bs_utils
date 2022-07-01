import sys
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

def make_bs_lst(n_bs, n_s, m_bs=None, seed=None, verbose=False):
    ''' Generate a list of integers to draw random samples of the data
        Args:
            - n_bs: the number of bootstrap samples to generate
            - n_s:   the number of original samples
            - m_bs: the number of random draws per bootstrap to generate
                    if m_bs != n_s, you will have to appropriately rescale
                    the fluctuations by sqrt( m_bs / n_s)
            - seed: a string that will be hashed to seed
                    the random number generator
        Return:
            bs_lst  a list of shape [n_bs, m_bs]
    '''

    # seed the random number generator
    rng = get_rng(seed,verbose=verbose) if seed else np.random.default_rng()

    if m_bs:
        Mbs = m_bs
    else:
        Mbs = n_s

    # make BS list: [low, high)
    return rng.integers(low=0, high=n_s, size=[n_bs, Mbs])

def bs_prior(n_bs, mean=0., sdev=1., seed=None, normal=True):
    ''' Generate bootstrap distribution of prior central values
        Args:
            n_bs : number of values to return
            mean : mean of Gaussian distribution
            sdev : width of Gaussian distribution
            seed : string to seed random number generator
        Return:
            a numpy array of length n_bs of normal(mean, sdev) values
    '''
    # seed the random number generator
    rng = get_rng(seed) if seed else np.random.default_rng()

    if normal:
        return rng.normal(loc=mean, scale=sdev, size=n_bs)
    else:
        sys.exit("Non-Gaussian distributions not currently supported")


def bs_corrs(corr, Nbs, Mbs=None, seed=None, return_bs_list=False, return_mbs=False, verbose=False):
    ''' WILL BE DEPRECATED SOON
        generate bootstrap resampling of correlation function data
        Args:
            - corr: numpy array of data (Ncfg, Nt, ...)
            - Nbs:  the number of bootstrap samples to generate
            - Mbs:  the number of random draws per bootstrap to generate
                    if Mbs != Ncfg, you will have to appropriately rescale
                    the fluctuations by sqrt( Mbs / Ncfg)
            - seed: a string that will be hashed to seed the random number generator

        Return:
            return_mbs=False
                corr_bs: an array of shape (Nbs, Nt, ...)
            return_mbs=True
                corr_bs: an array of shape (Nbs, Mbs, Nt, ...)
            return_bs_list=True
                corr_bs, bs_list.shape = (Nbs, Mbs)
    '''

    Ncfg = corr.shape[0]
    if Mbs:
        m_bs = Mbs
    else:
        m_bs = Ncfg

    # seed the random number generator
    rng = get_rng(seed,verbose=verbose) if seed else np.random.default_rng()

    # make BS list: [low, high)
    bs_list = rng.integers(low=0, high=Ncfg, size=[Nbs, m_bs])

    # make BS corrs
    corr_bs = np.zeros(tuple([Nbs, m_bs]) + corr.shape[1:], dtype=corr.dtype)
    for bs in range(Nbs):
        corr_bs[bs] = corr[bs_list[bs]]

    # if return_mbs, return (Nbs, Mbs, Nt, ...) array
    # otherwise, return mean over Mbs axis
    if return_mbs:
        bs_mean   = corr_bs.mean(axis=(0,1))
        d_corr_bs = corr_bs - bs_mean
        corr_bs   = bs_mean + d_corr_bs * np.sqrt( m_bs / Ncfg)
    else:
        corr_bs   = corr_bs.mean(axis=1)
        bs_mean   = corr_bs.mean(axis=0)
        d_corr_bs = corr_bs - bs_mean
        corr_bs   = bs_mean + d_corr_bs * np.sqrt( m_bs / Ncfg)

    if return_bs_list:
        return corr_bs, bs_list
    else:
        return corr_bs
