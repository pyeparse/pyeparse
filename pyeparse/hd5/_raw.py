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
            import tables
        except:
            raise ImportError('pytables is required but was not found')
        if not op.isfile(fname):
            raise IOError('file "%s" not found' % fname)
        info = dict()
        with tables.openFile(fname) as fid:
            # samples
            samples = fid.getNode('/', 'samples').read()
            info['sample_fields'] = list(deepcopy(samples.dtype.names))
            samples = samples.view(np.float64).reshape(samples.shape[0], -1).T
            # times
            times = fid.getNode('/', 'times').read()
            # discrete
            discrete = dict()
            dg = fid.getNode('/', 'discrete')
            for key in dg.__members__:
                discrete[key] = getattr(dg, key).read()
            # info
            data = fid.getNode('/', 'info').read()
            for key in data.dtype.names:
                info[key] = data[key][0]
            date = info['meas_date'].decode('ASCII')
            info['meas_date'] = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            # calibrations
            cg = fid.getNode(fid.root, 'calibrations')
            cals = np.array([fid.getNode(cg, 'c%s' % ii).read()
                             for ii in range(len(cg.__members__))])
            info['calibrations'] = cals

        self._samples = samples
        self._times = times
        self.discrete = discrete
        self.info = info
        _BaseRaw.__init__(self)  # perform sanity checks
