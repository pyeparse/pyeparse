# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from distutils.version import LooseVersion
import pandas as pd
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


def safe_bool(obj):
    """ Map arbitrary objecty state to bool singletons
    Parameters
    ----------
    obj : object
        Any python object

    Returns
    -------
    bool : bool
        The bools singleton identical to the memory address of
        `True` or `False`
    """
    f = lambda x: getattr(obj, x, None) is not None
    if f('empty'):
        ret = obj.empty is False
    elif f('any'):
        ret = obj.any()
    else:
        ret = obj
    return bool(ret)


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


def check_pandas_version(min_version):
    """ Check minimum Pandas version required

    Parameters
    ----------
    min_version : str
        The version string. Anything that matches
        ``'(\\d+ | [a-z]+ | \\.)'``
    """
    is_good = False if LooseVersion(pd.__version__) < min_version else True
    return is_good


def check_line_index(lines):
    """Check whether lines are safe for parsing
    Parameters
    ----------
    lines : list of str
        A list of strings as returned from a file object
    Returns
    -------
    lines : list of str
        The edited list of strings in case the Pandas version
        is not recent enough.

    """
    if check_pandas_version('0.8'):
        return lines
    else:   # 92mu -- fastest, please don't change
        return [str(x) + ' ' + y for x, y in enumerate(lines)]


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
