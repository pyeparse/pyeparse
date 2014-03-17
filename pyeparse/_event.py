# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
from ._py23 import string_types


class Discrete(list):
    """ Simple Container for discrete data based on Python list
    """

    def __init__(self, *args):
        list.__init__(self, *args)

    def __repr__(self):
        s = '<Discrete | {0} epochs; {1} events>'
        return s.format(len(self), sum(len(d) for d in self if d is not None))


def find_events(raw, pattern, event_id):
    """Find messages already parsed

    Parameters
    ----------
    raw : instance of pyeparse.raw.Raw
        the raw file to find events in.
    pattern : str | callable
        A substring to be matched or a callable that matches
        a string, for example ``lambda x: 'my-message' in x``
    event_id : int
        The event id to use.

    Returns
    -------
    idx : instance of numpy.ndarray (times, event_id)
        The indices found.
    """
    df = raw.discrete.get('messages', None)
    if df is not None:
        if callable(pattern):
            func = pattern
        elif isinstance(pattern, string_types):
            func = lambda x: pattern in x
        else:
            raise ValueError('Pattern not valid. Pass string or function')
        idx = np.array([func(msg) for msg in df['msg']])
        out = raw.time_as_index(df['time'][idx])
        id_vector = np.repeat(event_id, len(out)).astype(np.int64)
        return np.c_[out, id_vector]
    else:
        return np.zeros((0, 2), dtype=np.int64)
