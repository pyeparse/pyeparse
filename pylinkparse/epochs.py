# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pandas as pd
import copy
import numpy as np
from numpy.testing import assert_array_less
from .event import Discrete


class Epochs(object):
    """ Create epoched data

    Parameters
    ----------
    raw : instance of pylabparse.raw.Raw
        The raw instance to create epochs from
    events : ndarray (n_samples)
        The events to construct epochs around.
    tmin : float
        The time window before a particular event in seconds.
    tmax : float
        The time window after a particular event in seconds.
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

        discrete_inds = [{k: [] for k in my_event_id} for _ in '....']
        sample_inds = discrete_inds.pop(0)
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
                # import pdb;pdb.set_trace()
                stime, etime = df[['stime', 'etime']].values.T
                assert_array_less(stime, etime)
                event_in_window = np.where((stime >= this_tmin) &
                                           (etime <= this_tmax))
                parsed[this_id].append([event_in_window[0], ii])
            keep_idx.append(ii)
            ii += 1

        self.events = events[keep_idx]
        min_samples = np.min(min_samples)

        for kind, parsed in zip(['saccades', 'fixations', 'blinks'],
                                discrete_inds):
            this_in = raw.discrete.get(kind, None)
            if this_in is not None:
                for this_id, values in parsed.iteritems():
                    this_id = (this_id if event_keys is None else
                               event_keys[this_id])
                    this_discrete = Discrete()
                    for ind in zip(*values)[0]:
                        if ind.any().any():
                            df = this_in.iloc[ind]
                            df['event_id'] = this_id
                            this_discrete.append(df)
                        else:
                            this_discrete.append([])

                    setattr(self, kind + '_', this_discrete)

        _samples = []
        c = np.concatenate
        track_inds = []
        for this_id, values in sample_inds.iteritems():
            ind, _ = zip(*values)
            ind = [i[:min_samples] for i in ind]
            df = raw.samples.iloc[c(ind)]
            this_id = this_id if event_keys is None else event_keys[this_id]
            df['event_id'] = this_id
            count = c([np.repeat(v, min_samples) for _, v in values])
            df['epoch_idx'] = count
            _samples.append(df)
            track_inds.extend([len(i) for i in ind])

        sort_k = ['epoch_idx', 'time']  # important for multiple conditions
        # ignore index to allow for sorting + keep unique values
        self.data = pd.concat(_samples, ignore_index=True)
        self.data.sort(sort_k, inplace=True)
        assert set(track_inds) == set([min_samples])
        n_samples = min_samples
        n_epochs = len(track_inds)
        self.times = np.linspace(tmin, tmax, n_samples)
        self.data['times'] = np.tile(self.times, n_epochs)
        self._n_times = min_samples

        self.data.set_index(['epoch_idx', 'times'], drop=True,
                            inplace=True, verify_integrity=True)

    def __repr__(self):
        s = '<Epochs | {0} events | tmin: {1} tmax: {2}>'
        return s.format(len(self.events), self.tmin, self.tmax)

    def next(self):
        return NotImplemented

    def __iter__(self):
        return NotImplemented

    def __getitem__(self, idx):
        return NotImplemented
