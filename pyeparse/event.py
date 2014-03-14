# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import re
import numpy as np
from .utils import string_types, raw_open


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


def find_custom_events(raw, fname, pattern, event_id, prefix=True, sep='\W+',
                       return_residuals=False):
    """Find arbitrary messages from raw data file

    Parameters
    ----------
    raw : instance of pyeparse.raw.Raw
        the raw file to find events in.
    fname : str
        The filename for the data file to search. This must be provided
        because the data stored in ``raw`` has been converted from the
        original format (and thus cannot be traversed).
    pattern : str
        A substring to be matched (using regular expressions).
    event_id : int
        The event id to use.
    prefix : bool
        Whether the message includes a prefix, e.g., MSG or
        directly begins with the time sample.
    sep : str
        The separator (will be regex matched to split the line).
    return_residuals : bool
        If True, then the rest of the event line will be returned for
        each event.

    Returns
    -------
    idx : instance of numpy.ndarray
        The indices found.
    """
    events = []
    residuals = []

    idx = 1 if prefix else 0
    with raw_open(fname) as fid:
        for line in fid:
            if len(re.findall(pattern, line)) > 0:
                event = re.split(sep, line)
                residuals.append(event[idx+1:])
                event = event[idx]
                try:
                    event = float(event)
                except ValueError:
                    raise ValueError('could not convert to float: "%s", '
                                     'perhaps separator is incorrect?'
                                     % event)
                events.append(event)
    events = np.array(events, dtype='f8')
    events -= raw._t_zero
    events /= 1e3
    out = raw.time_as_index(events)
    out = np.c_[out, np.repeat(event_id, len(out))].astype(np.int64)
    if return_residuals:
        return out, residuals
    else:
        return out
