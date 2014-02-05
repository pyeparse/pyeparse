# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
from .utils import safe_bool, string_types


class Discrete(list):
    """ Simple Container for discrete data based on Python list
    """

    def __init__(self, *args):
        list.__init__(self, *args)

    def __repr__(self):
        s = '<Discrete | {0} epochs; {1} events>'
        return s.format(len(self), sum(len(d) for d in self if safe_bool(d)))


def find_events(raw, pattern, event_id):
    """Find messages already parsed

    Parameters
    ----------
    raw : instance of pylinkparse.raw.Raw
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
    if safe_bool(df):
        if callable(pattern):
            func = pattern
        elif isinstance(pattern, string_types):
            func = lambda x: pattern in x
        else:
            raise ValueError('Pattern not valid. Pass string or function')
        my_bool = df.msg.map(func)
        out = raw.time_as_index(df['time'].ix[my_bool.nonzero()[0]])
        id_vector = np.repeat(event_id, len(out)).astype(np.int64)
        return np.c_[out, id_vector]


def find_custom_events(raw, pattern, event_id, prefix=True, sep=' '):
    """Find arbitrary messages from raw data file

    Parameters
    ----------
    raw : instance of pylinkparse.raw.Raw
        the raw file to find events in.
    pattern : str
        A substring to be matched
    event_id : int
        The event id to use.
    prefix : bool
        Whether the message includes a prefix, e.g., MSG or
        directly begins with the time sample.
    sep : str
        The separator.

    Returns
    -------
    idx : instance of numpy.ndarray
        The indices found.
    """
    events = []

    idx = 1 if prefix else 0
    with open(raw.info['fname']) as fid:
        for line in fid:
            if pattern in line:
                events.append(line.split(sep)[idx])
    events = np.array(events, dtype='f8')
    events -= raw._t_zero
    events /= 1e3
    out = raw.time_as_index(events)
    return np.c_[out, np.repeat(event_id, len(out))].astype(np.int64)
