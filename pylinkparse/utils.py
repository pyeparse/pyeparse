# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)


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
