import numpy as np
from nose.tools import assert_raises
from functools import partial
from os import path as op
import warnings
import matplotlib

from pylinkparse import Raw, Epochs
from pylinkparse.viz import _epochs_axes_onclick, _epochs_navigation_onclick

warnings.simplefilter('always')  # in case we hit warnings
matplotlib.use('Agg')  # for testing don't use X server

path = op.join(op.dirname(__file__), 'data')
fnames = [op.join(path, 'test_raw.asc'),
          op.join(path, 'test_2_raw.asc')]


class DummyEvent(object):
    def __init__(self, ax):
        self.inaxes = ax


def test_raw_plot():
    """Test plotting of raw"""
    for fi, fname in enumerate(fnames):
        raw = Raw(fname)
        if 'calibration' in raw.info:
            raw.plot_calibration()
        else:
            assert_raises(RuntimeError, raw.plot_calibration)
        if 'screen_coords' in raw.info:
            raw.plot_heatmap(0., 10.)
        else:
            assert_raises(RuntimeError, raw.plot_heatmap)


def test_epochs_plot():
    """Test plotting of epochs"""
    tmin, tmax, event_id = -0.5, 1.5, 999
    # create some evil events
    events = np.array([[1000, 999], [2000, 999], [3000, 999]])
    for fname in fnames:
        raw = Raw(fname)
        epochs = Epochs(raw, events, event_id, tmin, tmax)
        assert_raises(ValueError, epochs.plot, picks=['whatever'])
        epochs.plot(picks=['ps'])
        fig = epochs.plot(n_chunks=2)

    # test clicking: find our callbacks
    for func in fig.canvas.callbacks.callbacks['button_press_event'].items():
        func = func[1].func
        if isinstance(func, partial):
            break
    params = func.keywords['params']
    for ax in fig.axes:
        _epochs_axes_onclick(DummyEvent(ax), params)

    # nav clicking
    _epochs_navigation_onclick(DummyEvent(params['next'].ax), params)
    _epochs_navigation_onclick(DummyEvent(params['back'].ax), params)
    _epochs_navigation_onclick(DummyEvent(params['reject-quit'].ax), params)
