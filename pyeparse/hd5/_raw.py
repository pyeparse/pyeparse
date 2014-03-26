# -*- coding: utf-8 -*-
"""HD5 Raw class"""

import numpy as np
from os import path as op
from copy import deepcopy
from datetime import datetime

from .._baseraw import _BaseRaw


class RawHD5(_BaseRaw):
    """Represent HD5 files in Python

    Parameters
    ----------
    fname : str
        The name of the EDF file.
    """
    def __init__(self, fname):
        try:
            import tables as tb
        except:
            raise ImportError('pytables is required but was not found')
        if not op.isfile(fname):
            raise IOError('file "%s" not found' % fname)
        info = dict()
        o_f = tb.open_file if hasattr(tb, 'open_file') else tb.openFile
        with o_f(fname) as fid:
            # samples
            g_n = fid.get_node if hasattr(fid, 'get_node') else fid.getNode
            samples = g_n('/', 'samples').read()
            info['sample_fields'] = list(deepcopy(samples.dtype.names))
            samples = samples.view(np.float64).reshape(samples.shape[0], -1).T
            # times
            times = g_n('/', 'times').read()
            # discrete
            discrete = dict()
            dg = g_n('/', 'discrete')
            for key in dg.__members__:
                discrete[key] = getattr(dg, key).read()
            # info
            data = g_n('/', 'info').read()
            for key in data.dtype.names:
                info[key] = data[key][0]
            date = info['meas_date'].decode('ASCII')
            info['meas_date'] = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            # calibrations
            cg = g_n(fid.root, 'calibrations')
            cals = np.array([g_n(cg, 'c%s' % ii).read()
                             for ii in range(len(cg.__members__))])
            info['calibrations'] = cals

        self._samples = samples
        self._times = times
        self.discrete = discrete
        self.info = info
        _BaseRaw.__init__(self)  # perform sanity checks
