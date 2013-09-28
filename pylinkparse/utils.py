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
