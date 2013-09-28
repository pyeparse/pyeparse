import numpy as np
from os import path as op
from numpy.testing import assert_equal
from nose.tools import assert_true
from pylinkparse import Raw, Epochs
import glob

fnames = glob.glob(op.join(op.split(__file__)[0], 'data', '*raw.asc'))


def test_epochs_io():
    """Test epochs IO functionality"""

    tmin, tmax, event_id = -0.5, 1.5, 999
    events = np.array([[1000, 999], [10000, 999], [12000, 77]])
    for fname in fnames:
        raw = Raw(fname)
        epochs = Epochs(raw, events, event_id, tmin, tmax)
        print epochs  # test repr works
        for disc in epochs.info['discretes']:
            assert_equal(len(vars(epochs)[disc]), len(epochs.events))
        assert_equal(len(epochs.events), 2)
        assert_equal(epochs.data_frame.shape[0] / epochs._n_times,
                     len(epochs.events))

        epochs = Epochs(raw, events, dict(a=999, b=77), tmin, tmax)
        assert_equal(len(epochs.events), 3)
        assert_equal(epochs.data_frame.shape[0] / epochs._n_times,
                     len(epochs.events))

        for disc in epochs.info['discretes']:
            assert_equal(len(vars(epochs)[disc]), len(epochs.events))

        epochs2 = epochs.copy()
        assert_true(epochs._data is not epochs2._data)
        del epochs2._data
        assert_true('_data' in vars(epochs) and
                    '_data' not in vars(epochs2))
        assert_true(epochs is not epochs2)
        epochs2 = epochs[0]
        assert_equal(len(epochs2.events), 1)
        assert_equal(set(epochs2.events[:, -1]), {999})
        desired = len(epochs2.events) * len(epochs.times)
        assert_equal(epochs2.data_frame.shape[0], desired)

        epochs2 = epochs[[0, 1]]
        assert_equal(len(epochs2.events), 2)
        assert_equal(set(epochs2.events[:, -1]), {999})
        desired = len(epochs2.events) * len(epochs.times)
        assert_equal(epochs2.data_frame.shape[0], desired)

        epochs2 = epochs['a']
        assert_equal(len(epochs2.events), 2)
        assert_equal(set(epochs2.events[:, -1]), {999})
        desired = len(epochs2.events) * len(epochs.times)
        assert_equal(epochs2.data_frame.shape[0], desired)

        epochs2 = epochs[slice(1, 3)]
        assert_equal(len(epochs2.events), 2)
        assert_equal(set(epochs2.events[:, -1]), {999, 77})
        desired = len(epochs2.events) * len(epochs.times)
        assert_equal(epochs2.data_frame.shape[0], desired)
