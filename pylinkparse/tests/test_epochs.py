import numpy as np
from os import path as op
from numpy.testing import assert_equal
from pylinkparse import Raw, Epochs

fname = op.join(op.split(__file__)[0], 'data', 'test_raw.asc')


def test_epochs_io():
    """Test epochs IO functionality"""
    tmin, tmax, event_id = -0.5, 1.5, 999
    events = np.array([[1000, 999], [10000, 999], [12000, 77]])
    raw = Raw(fname)
    epochs = Epochs(raw, events, event_id, tmin, tmax)
    print epochs  # test repr works
    assert_equal(len(epochs.events), 2)
    assert_equal(epochs.data_frame.shape[0] / epochs._n_times,
                 len(epochs.events))

    epochs = Epochs(raw, events, dict(a=999, b=77), tmin, tmax)
    assert_equal(len(epochs.events), 3)
    assert_equal(epochs.data_frame.shape[0] / epochs._n_times,
                 len(epochs.events))

    epochs[0]
    epochs[[0, 1]]
    epochs['a']
    epochs[slice(2)]
