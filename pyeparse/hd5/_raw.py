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
            import h5py
        except:
            raise ImportError('h5py is required but was not found')
        if not op.isfile(fname):
            raise IOError('file "%s" not found' % fname)
        info = dict()
        with h5py.File(fname, mode='r') as fid:
            # samples
            samples = np.array(fid['samples'])
            info['sample_fields'] = list(deepcopy(samples.dtype.names))
            samples = samples.view(np.float64).reshape(samples.shape[0], -1).T
            # times
            times = np.array(fid['times'])
            # discrete
            discrete = dict()
            dg = fid['discrete']
            for key in dg.keys():
                discrete[key] = np.array(dg[key])
            # info
            data = np.array(fid['info'])
            for key in data.dtype.names:
                info[key] = data[key][0]
            date = info['meas_date'].decode('ASCII')
            info['meas_date'] = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            # calibrations
            cg = fid['calibrations']
            cals = np.array([np.array(cg['c%s' % ii])  # maintain order
                             for ii in range(len(cg.keys()))])
            info['calibrations'] = cals

        self._samples = samples
        self._times = times
        self.discrete = discrete
        self.info = info
        _BaseRaw.__init__(self)  # perform sanity checks
