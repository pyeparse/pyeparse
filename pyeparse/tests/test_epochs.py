import numpy as np
import warnings
from numpy.testing import assert_equal, assert_array_equal
from nose.tools import assert_true, assert_raises

from pyeparse import Raw, Epochs
from pyeparse.utils import _get_test_fnames, _has_joblib, _requires_edfapi

warnings.simplefilter('always')  # in case we hit warnings

fnames = _get_test_fnames()


def _filter_warnings(w):
    return [ww for ww in w if 'Did not find event' in str(ww)]


@_requires_edfapi
def test_epochs_deconv():
    """Test epochs deconvolution"""
    tmin, tmax = -0.5, 1.5
    event_dict = dict(foo=999)
    events = np.array([np.arange(0, 21000, 1000, int),
                       999 * np.ones(21, int)]).T
    for fi, fname in enumerate(fnames):
        if fi == 0:
            n_jobs = 1
        else:
            n_jobs = 0
        raw = Raw(fname)
        epochs = Epochs(raw, events, event_dict,
                        tmin, tmax)
        a = raw.info['sample_fields']
        b = epochs.info['data_cols']
        assert_equal(len(a), len(b))
        assert_true(all(aa == bb for aa, bb in zip(a, b)))
        data = epochs.get_data('ps')
        assert_raises(RuntimeError, Epochs, raw, events, 'test', tmin, tmax)
        fit, times = epochs.deconvolve()
        assert_array_equal(data, epochs.get_data('ps'))
        assert_equal(fit.shape, (len(epochs), len(times)))
        fit, times = epochs.deconvolve(spacing=[-0.1, 0.4, 1.0],
                                       bounds=(0, np.inf), n_jobs=n_jobs)
        assert_equal(fit.shape, (len(epochs), len(times)))
        assert_equal(len(times), 3)
        if fi == 0:
            if _has_joblib():
                assert_raises(ValueError, epochs.deconvolve, n_jobs=-1000)


@_requires_edfapi
def test_epochs_combine():
    """Test epochs combine IDs functionality"""
    tmin, tmax = -0.5, 1.5
    event_dict = dict(foo=1, bar=2, test=3)
    events_1 = np.array([[12000, 1], [1000, 2], [10000, 2], [2000, 3]])
    events_2 = np.array([[12000, 2], [1000, 1], [10000, 1], [2000, 3]])
    for fname in fnames:
        raw = Raw(fname)
        epochs_1 = Epochs(raw, events_1, event_dict, tmin, tmax)
        epochs_2 = Epochs(raw, events_2, event_dict, tmin, tmax)
        assert_raises(ValueError, epochs_1.combine_event_ids, ['foo', 'bar'],
                      dict(foobar=1))
        epochs_1.combine_event_ids(['foo', 'bar'], 12)
        epochs_2.combine_event_ids(['foo', 'bar'], dict(foobar=12))
        d1 = epochs_1.data
        d2 = epochs_2.data
        assert_array_equal(d1, d2)

        epochs_1.equalize_event_counts(['12', 'test'])
        epochs_2.equalize_event_counts(['foobar', 'test'])
        d1 = epochs_1.data
        d2 = epochs_2.data
        assert_array_equal(d1, d2)
        # this shouldn't really do anything
        epochs_2.equalize_event_counts(['foobar', 'test'], method='truncate')
        assert_array_equal(d1, epochs_2.data)


@_requires_edfapi
def test_epochs_concat():
    """Test epochs concatenation"""
    tmin, tmax = -0.5, 1.5
    event_dict = dict(foo=999, bar=77)
    events_a = np.array([[12000, 77], [1000, 999], [-1, 999]])
    events_b = np.array([[1000, 999], [10000, 999]])
    for fname in fnames:
        raw = Raw(fname)
        events_a[-1, 0] = raw.n_samples - 1
        epochs_ab = Epochs([raw] * 2, [events_a, events_b], event_dict,
                           tmin, tmax)
        epochs_ba = Epochs([raw] * 2, [events_b, events_a], event_dict,
                           tmin, tmax)
        # make sure discretes made it through
        for epochs in [epochs_ab, epochs_ba]:
            for d in [epochs.blinks, epochs.saccades, epochs_ab.fixations]:
                assert_equal(len(d), len(epochs))
                for dd in d:
                    if len(dd) > 0:
                        for t in (dd['stime'], dd['etime']):
                            assert_true(np.all(t >= tmin) & np.all(t <= tmax))
        assert_equal(len(epochs_ab.events), 4)
        assert_equal(len(epochs_ba.events), 4)
        assert_array_equal(epochs_ab.times, epochs_ba.times)
        # make sure event numbers match
        reord = [2, 3, 0, 1]
        assert_array_equal(epochs_ab.events[:, 1], epochs_ba.events[reord, 1])
        # make sure actual data matches
        data_ab = epochs_ab.data
        data_ba = epochs_ba.data
        assert_array_equal(data_ab, data_ba[reord])
        # test discretes
        assert_equal(len(epochs_ab.blinks), len(epochs_ba.blinks))
        blink_ab = epochs_ab.blinks[3]
        blink_ba = epochs_ba.blinks[reord[3]]
        assert_equal(len(blink_ab), len(blink_ba))
        assert_true(len(blink_ab) > 0)  # make sure we've tested useful case
        for key in ('stime', 'etime'):
            blink_ab_d = blink_ab[key]
            blink_ba_d = blink_ba[key]
            assert_array_equal(blink_ab_d, blink_ba_d)


@_requires_edfapi
def test_epochs_io():
    """Test epochs IO functionality"""
    tmin, tmax, event_id = -0.5, 1.5, 999
    missing_event_dict = dict(foo=999, bar=555)
    # create some evil events
    events = np.array([[12000, 77], [1000, 999], [10000, 999]])
    for fname in fnames:
        raw = Raw(fname)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            epochs = Epochs(raw, events, missing_event_dict, tmin, tmax,
                            ignore_missing=True)
        assert_raises(RuntimeError, Epochs, raw, events, 1.1, tmin, tmax)
        assert_raises(ValueError, Epochs, [raw] * 2, events, event_id,
                      tmin, tmax)
        assert_equal(len(_filter_warnings(w)), 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            epochs = Epochs(raw, events, missing_event_dict, tmin, tmax)
        assert_equal(len(_filter_warnings(w)), 1)
        epochs = Epochs(raw, events, event_id, tmin, tmax)
        assert_raises(IndexError, epochs.drop_epochs, [1000])
        print(epochs)  # test repr works
        for disc in epochs.info['discretes']:
            assert_equal(len(vars(epochs)[disc]), len(epochs.events))
        assert_equal(len(epochs.events), 2)
        # assert_equal(epochs.data_frame.shape[0] / epochs._n_times,
        #              len(epochs.events))
        # assert_true(epochs.data_frame['time'].diff().min() >= 0)

        epochs = Epochs(raw, events, dict(a=999, b=77), tmin, tmax)
        assert_equal(len(epochs.events), 3)
        # assert_equal(epochs.data_frame.shape[0] / epochs._n_times,
        #              len(epochs.events))
        # assert_true(epochs.data_frame['time'].diff().min() >= 0)

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
        # desired = len(epochs2.events) * len(epochs.times)
        # assert_equal(epochs2.data_frame.shape[0], desired)
        # assert_true(epochs2.data_frame['time'].diff().min() >= 0)
        assert_equal(len(epochs2.saccades), len(epochs2.events))

        epochs2 = epochs[[1, 0]]
        assert_equal(len(epochs2.events), 2)
        assert_equal(set(epochs2.events[:, -1]), set([999]))
        # desired = len(epochs2.events) * len(epochs.times)
        assert_equal(len(epochs2.saccades), len(epochs2.events))
        # assert_equal(epochs2.data_frame.shape[0], desired)

        epochs2 = epochs['a']
        assert_equal(len(epochs2.events), 2)
        assert_equal(set(epochs2.events[:, -1]), set([999]))
        # desired = len(epochs2.events) * len(epochs.times)
        # assert_equal(epochs2.data_frame.shape[0], desired)
        assert_equal(len(epochs2.saccades), len(epochs2.events))

        epochs2 = epochs[['a', 'b']]
        assert_equal(len(epochs2.events), 3)
        assert_equal(set(epochs2.events[:, -1]), set([999, 77]))
        # desired = len(epochs2.events) * len(epochs.times)
        # assert_equal(epochs2.data_frame.shape[0], desired)
        assert_equal(len(epochs2.saccades), len(epochs2.events))
        assert_true(np.diff(epochs2.events[:, 0]).min() >= 0)

        epochs2 = epochs[slice(1, 3)]
        assert_equal(len(epochs2.events), 2)
        assert_equal(set(epochs2.events[:, -1]), set([999, 77]))
        # desired = len(epochs2.events) * len(epochs.times)
        # assert_equal(epochs2.data_frame.shape[0], desired)
        assert_equal(len(epochs2.saccades), len(epochs2.events))
        assert_true(np.diff(epochs2.events[:, 0]).min() >= 0)
        """
        data1 = epochs[0].data
        data2 = epochs.data_frame.ix[0, epochs.info['data_cols']].values
        data2 = data2.reshape(1,
                              len(epochs.times),
                              len(epochs.info['data_cols']))
        assert_array_equal(data1, np.transpose(data2, [0, 2, 1]))
        """

        for e in epochs:
            assert_true(np.argmin(e.shape) == 0)
        assert_array_equal(e, epochs.data[-1])
