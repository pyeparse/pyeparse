# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np


class Discrete(list):
    """ Simple Container for discrete data based on Python list
    """

    def __init__(self):
        pass

    def __repr__(self):
        s = '<Discrete | {0} epochs; {1} events>'
        return s.format(len(self), sum(len(d) for d in self if d))


def find_custom_events(raw, pattern, event_id, prefix=True, sep=' '):
    """Find arbitrary messages
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
    out = np.nonzero(np.in1d(raw.samples['time'], events))[0]
    return np.c_[out, np.repeat(event_id, len(out))]
