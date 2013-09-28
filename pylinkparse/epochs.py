# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pandas as pd
import copy
from functools import wraps
import numpy as np
from numpy.testing import assert_array_less
from .event import Discrete


class Epochs(object):
    """ Create epoched data

    Parameters
    ----------
    raw : instance of pylabparse.raw.Raw
        The raw instance to create epochs from
    events : ndarray (n_epochs)
        The events to construct epochs around.
    event_id : int
        The event ID to use.
    tmin : float
        The time window before a particular event in seconds.
    tmax : float
        The time window after a particular event in seconds.

    Returns
    -------
    epochs : instance of Epochs
        The epoched dataset.
    """
    def __init__(self, raw, events, event_id, tmin, tmax):
        self.info = copy.deepcopy(raw.info)
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        data, times = raw[:]
        event_keys = None
        if isinstance(event_id, dict):
            my_event_id = event_id.values()
            event_keys = {v: k for k, v in event_id.items()}
        elif np.isscalar(event_id):
            my_event_id = [event_id]

        discrete_inds = [[] for _ in range(3)]
        sample_inds = {k: [] for k in my_event_id}
        saccade_inds, fixation_inds, blink_inds = discrete_inds
        keep_idx = []
        ii = 0
        min_samples = []
        for event, this_id in events:
            if this_id not in my_event_id:
                continue
            this_time = times[event]
            this_tmin, this_tmax = this_time + tmin, this_time + tmax
            inds_min, inds_max = raw.time_as_index([this_tmin, this_tmax])
            if max([inds_min, inds_max]) >= len(raw.samples):
                break
            inds = np.arange(inds_min, inds_max)
            min_samples.append(inds.shape[0])

            sample_inds[this_id].append([inds, ii])
            for kind, parsed in zip(raw.info['event_types'], discrete_inds):
                df = raw.discrete.get(kind, kind)
                assert_array_less(df['stime'], df['etime'])
                event_in_window = np.where((df['stime'] >= this_tmin) &
                                           (df['etime'] <= this_tmax))
                parsed.append([event_in_window[0], ii, this_id])
            keep_idx.append(ii)
            ii += 1

        self.events = events[keep_idx]
        min_samples = np.min(min_samples)

        for kind, parsed in zip(['saccades', 'fixations', 'blinks'],
                                discrete_inds):
            this_in = raw.discrete.get(kind, None)
            if this_in is not None:
                this_discrete = Discrete()
                for inds, epochs_idx, this_id in parsed:
                    this_id = (this_id if event_keys is None else
                               event_keys[this_id])
                    if inds.any().any():
                        df = this_in.ix[inds]
                        df['event_id'] = this_id
                        this_discrete.append(df)
                    else:
                        this_discrete.append([])
                this_name = kind + '_'
                setattr(self, this_name, this_discrete)
                self.info['discretes'] += [this_name]

        _samples = []
        c = np.concatenate
        track_inds = []
        for this_id, values in sample_inds.iteritems():
            ind, _ = zip(*values)
            ind = [i[:min_samples] for i in ind]
            df = raw.samples.ix[c(ind)]
            this_id = this_id if event_keys is None else event_keys[this_id]
            df['event_id'] = this_id
            count = c([np.repeat(vv, min_samples) for _, vv in values])
            df['epoch_idx'] = count
            _samples.append(df)
            track_inds.extend([len(i) for i in ind])

        sort_k = ['epoch_idx', 'time']  # important for multiple conditions
        # ignore index to allow for sorting + keep unique values
        self._data = pd.concat(_samples, ignore_index=True)
        self._data.sort(sort_k, inplace=True)
        assert set(track_inds) == set([min_samples])
        n_samples = min_samples
        n_epochs = len(track_inds)
        self.times = np.linspace(tmin, tmax, n_samples)
        self._data['times'] = np.tile(self.times, n_epochs)
        self._n_times = min_samples
        self._n_epochs = n_epochs

        self._data.set_index(['epoch_idx', 'times'], drop=True,
                             inplace=True, verify_integrity=True)
        self._current = 0

    def __repr__(self):
        s = '<Epochs | {0} events | tmin: {1} tmax: {2}>'
        return s.format(len(self.events), self.tmin, self.tmax)

    def __iter__(self):
        """To make iteration over epochs easy.
        """
        self._current = 0
        return self

    def next(self, return_event_id=False):
        """To make iteration over epochs easy.
        """

        if self._current >= self._n_epochs:
            raise StopIteration
        epoch = self.data_frame.ix[self._current]
        epoch = epoch[self.info['data_cols']].values
        self._current += 1
        if not return_event_id:
            return epoch
        else:
            return epoch, self.events[self._current - 1][-1]

    @property
    def data(self):
        out = self._data[self.info['data_cols']].values
        out = out.reshape(len(self.events),
                          len(self.times),
                          len(self.info['data_cols']))
        return np.transpose(out, [0, 2, 1])

    @property
    def data_frame(self):
        return self._data

    def __getitem__(self, idx):
        out = self.copy()
        if isinstance(idx, basestring):
            if idx not in self.event_id:
                raise ValueError('ID not found')
            idx = self.event_id[idx]
            idx = np.where(self.events[:, -1] == idx)[0]
        elif np.isscalar(idx):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = np.arange(*idx.indices(idx.stop))

        midx = [i for i in out._data.index if i[0] in idx]
        out._data = out._data.ix[midx]
        out.events = out.events[idx]
        for discrete in self.info['discretes']:
            disc = vars(self)[discrete]
            disc = Discrete(disc[k] for k in idx)
        return out

    def copy(self):
        """Return a copy of Epochs.
        """
        return copy.deepcopy(self)
