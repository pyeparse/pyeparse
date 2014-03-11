# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from distutils.version import LooseVersion
import numpy as np


# Store these in one place
discrete_types = ['saccades', 'fixations', 'blinks']


# For python3
try:
    advance_iterator = next
except NameError:
    def advance_iterator(it):
        return it.next()
next = advance_iterator


try:
    string_types = basestring  # noqa
except NameError:
    string_types = str


def create_chunks(sequence, size):
    """Generate chunks from a sequence

    Note. copied from MNE-Python

    Parameters
    ----------
    sequence : iterable
        Any iterable object
    size : int
        The chunksize to be returned
    """
    return (sequence[p:p + size] for p in range(0, len(sequence), size))


def check_pandas_version(min_version='0.0'):
    """ Check minimum Pandas version required

    Parameters
    ----------
    min_version : str
        The version string. Anything that matches
        ``'(\\d+ | [a-z]+ | \\.)'``
    """
    min_version = LooseVersion(min_version)
    try:
        import pandas as pd
    except:
        return False
    is_good = False if LooseVersion(pd.__version__) < min_version else True
    return is_good


requires_pandas = np.testing.dec.skipif(not check_pandas_version(),
                                        'Requires Pandas')


def fwhm_kernel_2d(size, fwhm, center=None):
    """ Make a square gaussian kernel.

    Note: adapted from https://gist.github.com/andrewgiessel/4635563

    Parameters
    ----------
    size : int
        The length of the square matrix to create.
    fmhw : int
        The full wdith at hald maximum value.

    """
    x = np.arange(0, size, 1, np.float64)
    y = x[:, np.newaxis]
    # center
    x0 = y0 = size // 2

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def pupil_kernel(fs, dur=4.0):
    """Canonical pupil response kernel"""
    n_samp = int(np.round(fs * dur))
    t = np.arange(n_samp, dtype=float) / fs
    n = 10.1
    t_max = 0.930
    h = (t ** n) * np.exp(- n * t / t_max)
    h = 0.015 * h / (np.sum(h) * (t[1] - t[0]))
    return h
