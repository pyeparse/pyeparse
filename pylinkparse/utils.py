# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from distutils.version import LooseVersion
import pandas as pd


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
    return (sequence[p:p + size] for p in xrange(0, len(sequence), size))


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
