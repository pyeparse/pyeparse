# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
from os import path as op
import shutil
import tempfile
import subprocess

from ._py23 import string_types

# Store these in one place
discrete_types = ['saccades', 'fixations', 'blinks']


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


def pupil_kernel(fs, dur=4.0):
    """Canonical pupil response kernel"""
    n_samp = int(np.round(fs * dur))
    t = np.arange(n_samp, dtype=float) / fs
    n = 10.1
    t_max = 0.930
    h = (t ** n) * np.exp(- n * t / t_max)
    h = 0.015 * h / (np.sum(h) * (t[1] - t[0]))
    return h


class raw_open(object):
    """Context manager that will convert EDF to ASC on the fly"""
    def __init__(self, fname):
        if not isinstance(fname, string_types):
            raise TypeError('fname must be a string')
        if not op.isfile(fname):
            raise IOError('File not found: %s' % fname)
        if fname.endswith('.edf'):
            # Ideally we will eventually handle the binary files directly
            out_dir = tempfile.mkdtemp('edf2asc')
            out_fname = op.join(out_dir, 'temp.asc')
            p = subprocess.Popen(['edf2asc', fname, out_fname],
                                 stderr=subprocess.PIPE,
                                 stdout=subprocess.PIPE)
            stdout_, stderr = p.communicate()
            if p.returncode != 255:
                print((p.returncode, stdout_, stderr))
                raise RuntimeError('Could not convert EDF to ASC')
            self.fname = out_fname
            self.dir = out_dir
        else:
            self.fname = fname
            self.dir = None

    def __enter__(self):
        self.fid = open(self.fname, 'r')
        return self.fid

    def __exit__(self, type, value, traceback):
        self.fid.close()
        if self.dir is not None:
            shutil.rmtree(self.dir)


def _has_edf2asc():
    """See if the user has edf2asc"""
    p = subprocess.Popen(['edf2asc', '--help'],
                         stderr=subprocess.PIPE,
                         stdout=subprocess.PIPE)
    try:
        stdout_, stderr = p.communicate()
    except Exception:
        out = False
    else:
        out = True
    return out


def _get_test_fnames():
    """Get usable test files (omit EDF if no edf2asc)"""
    path = op.join(op.dirname(__file__), 'tests', 'data')
    fnames = [op.join(path, 'test_raw.asc')]
    if _has_edf2asc():
        fnames.append(op.join(path, 'test_2_raw.edf'))
    return fnames
