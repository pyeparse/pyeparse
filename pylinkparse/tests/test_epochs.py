import numpy as np
from os import path as op
from numpy.testing import assert_equal, assert_array_equal
from nose.tools import assert_true
import glob

from pylinkparse import Raw, Epochs

fnames = glob.glob(op.join(op.split(__file__)[0], 'data', '*raw.asc'))


def test_epochs_io():
    """Test epochs IO functionality"""

    tmin, tmax, event_id = -0.5, 1.5, 999
    event_dict = dict(foo=999, bar=77)
    # create some evil events
    events = np.array([[12000, 77], [1000, 999], [10000, 999]])
    for fname in fnames:
        raw = Raw(fname)
        epochs = Epochs([raw] * 3, [events] * 3, event_id, tmin, tmax)
        epochs = Epochs(raw, events, event_dict, tmin, tmax)
        epochs = Epochs(raw, events, event_id, tmin, tmax)
        print(epochs)  # test repr works
        for disc in epochs.info['discretes']:
            assert_equal(len(vars(epochs)[disc]), len(epochs.events))
        assert_equal(len(epochs.events), 2)
        assert_equal(epochs.data_frame.shape[0] / epochs._n_times,
                     len(epochs.events))
        assert_true(epochs.data_frame['time'].diff().min() >= 0)

        epochs = Epochs(raw, events, dict(a=999, b=77), tmin, tmax)
        assert_equal(len(epochs.events), 3)
        assert_equal(epochs.data_frame.shape[0] / epochs._n_times,
                     len(epochs.events))
        assert_true(epochs.data_frame['time'].diff().min() >= 0)

        for disc in epochs.info['discretes']:
            this_disc = vars(epochs)[disc]
            assert_equal(len(this_disc), len(epochs.events))
            for field in ['stime', 'etime']:
                for di in this_disc:
                    if field in di:
                        for event in di[field]:
                            assert_true(epochs.tmin <= event <= epochs.tmax)

        epochs2 = epochs.copy()
        assert_true(epochs._data is not epochs2._data)
        del epochs2._data
        assert_true('_data' in vars(epochs) and
                    '_data' not in vars(epochs2))
        assert_true(epochs is not epochs2)
        epochs2 = epochs[0]
        assert_equal(len(epochs2.events), 1)
        assert_equal(set(epochs2.events[:, -1]), set([999]))
        desired = len(epochs2.events) * len(epochs.times)
        assert_equal(epochs2.data_frame.shape[0], desired)
        assert_equal(len(epochs2.saccades_), len(epochs2.events))
        assert_true(epochs2.data_frame['time'].diff().min() >= 0)

        epochs2 = epochs[[1, 0]]
        assert_equal(len(epochs2.events), 2)
        assert_equal(set(epochs2.events[:, -1]), set([999]))
        desired = len(epochs2.events) * len(epochs.times)
        assert_equal(epochs2.data_frame.shape[0], desired)
        assert_equal(len(epochs2.saccades_), len(epochs2.events))

        epochs2 = epochs['a']
        assert_equal(len(epochs2.events), 2)
        assert_equal(set(epochs2.events[:, -1]), set([999]))
        desired = len(epochs2.events) * len(epochs.times)
        assert_equal(epochs2.data_frame.shape[0], desired)
        assert_equal(len(epochs2.saccades_), len(epochs2.events))

        epochs2 = epochs[['a', 'b']]
        assert_equal(len(epochs2.events), 3)
        assert_equal(set(epochs2.events[:, -1]), set([999, 77]))
        desired = len(epochs2.events) * len(epochs.times)
        assert_equal(epochs2.data_frame.shape[0], desired)
        assert_equal(len(epochs2.saccades_), len(epochs2.events))
        assert_true(np.diff(epochs2.events[:, 0]).min() >= 0)

        epochs2 = epochs[slice(1, 3)]
        assert_equal(len(epochs2.events), 2)
        assert_equal(set(epochs2.events[:, -1]), set([999, 77]))
        desired = len(epochs2.events) * len(epochs.times)
        assert_equal(epochs2.data_frame.shape[0], desired)
        assert_equal(len(epochs2.saccades_), len(epochs2.events))
        assert_true(np.diff(epochs2.events[:, 0]).min() >= 0)

        data1 = epochs[0].data
        data2 = epochs.data_frame.ix[0, epochs.info['data_cols']].values
        data2 = data2.reshape(1,
                              len(epochs.times),
                              len(epochs.info['data_cols']))
        assert_array_equal(data1, np.transpose(data2, [0, 2, 1]))

        for e in epochs:
            assert_true(np.argmin(e.shape) == 0)
        assert_array_equal(e, epochs.data[-1])
