import numpy as np
from nose.tools import assert_raises
from os import path as op

from pylinkparse import Raw, Epochs

path = op.join(op.split(__file__)[0], 'data')
fnames = [op.join(path, 'test_raw.asc'),
          op.join(path, 'test_2_raw.asc')]

import matplotlib
matplotlib.use('Agg')  # for testing don't use X server


def test_raw_plot():
    """Test plotting of raw"""
    for fi, fname in enumerate(fnames):
        raw = Raw(fname)
        if 'calibration' in raw.info:
            raw.plot_calibration()
        else:
            assert_raises(RuntimeError, raw.plot_calibration)
        if 'screen_coords' in raw.info:
            raw.plot_heatmap(0, 10)
        else:
            assert_raises(RuntimeError, raw.plot_heatmap)


def test_epochs_plot():
    """Test plotting of epochs"""
    tmin, tmax, event_id = -0.5, 1.5, 999
    # create some evil events
    events = np.array([[12000, 77], [1000, 999], [10000, 999]])
    for fname in fnames:
        raw = Raw(fname)
        epochs = Epochs(raw, events, event_id, tmin, tmax)
        epochs.plot()
