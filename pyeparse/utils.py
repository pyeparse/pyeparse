# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from os import path as op
import glob
import tempfile
from shutil import rmtree
import atexit


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


def pupil_kernel(fs, dur=4.0, t_max=0.930, n=10.1, s=1.):
    """Generate pupil response kernel modeled as an Erlang gamma function.

    Parameters
    ----------
    fs : int
        Sampling frequency (samples/second) to use in generating the kernel.
    dur : float
        Length (in seconds) of the generated kernel.
    t_max : float
        Time (in seconds) where the response maximum is stipulated to occur.
    n : float
        Number of negative-exponential layers in the cascade defining the
    s : float | None
        Desired value for the area under the kernel. If `None`, no scaling is
        performed.
    """

    n_samp = int(np.round(fs * dur))
    t = np.arange(n_samp, dtype=float) / fs
    h = (t ** n) * np.exp(- n * t / t_max)
    scal = 1. if s is None else float(s) / np.sum(h) * (t[1] - t[0])
    h = scal * h
    return h


def _get_test_fnames():
    """Get usable test files (omit EDF if no edf2asc)"""
    path = op.join(op.dirname(__file__), 'tests', 'data')
    fnames = glob.glob(op.join(path, '*.edf'))
    return fnames


class _TempDir(str):
    """Class for creating and auto-destroying temp dir

    This is designed to be used with testing modules.

    We cannot simply use __del__() method for cleanup here because the rmtree
    function may be cleaned up before this object, so we use the atexit module
    instead.
    """
    def __new__(self):
        new = str.__new__(self, tempfile.mkdtemp())
        return new

    def __init__(self):
        self._path = self.__str__()
        atexit.register(self.cleanup)

    def cleanup(self):
        rmtree(self._path, ignore_errors=True)


def _has_joblib():
    """Helper to determine if joblib is installed"""
    try:
        import joblib  # noqa
    except Exception:
        return False
    else:
        return True


def _has_h5py():
    """Helper to determine if joblib is installed"""
    try:
        import h5py  # noqa
    except Exception:
        return False
    else:
        return True


def _has_edfapi():
    """Helper to determine if a user has edfapi installed"""
    from .edf._raw import has_edfapi
    return has_edfapi

_requires_h5py = np.testing.dec.skipif(not _has_h5py(),
                                       'Requires h5py')


_requires_edfapi = np.testing.dec.skipif(not _has_edfapi(), 'Requires edfapi')
