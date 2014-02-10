# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pandas as pd
import copy
import numpy as np
import warnings
from .event import Discrete
from .viz import plot_epochs
from .utils import string_types


class Epochs(object):
    """ Create epoched data

    Parameters
    ----------
    raw : instance of Raw | list
        The raw instance to create epochs from. Can also be a list of raw
        instances to use.
    events : ndarray (n_epochs) | list
        The events to construct epochs around. Can also be a list of
        arrays.
    event_id : int | dict
        The event ID to use. Can be a dict to supply multiple event types
        by name.
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
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self._current = 0
        event_keys = None
        if not isinstance(raw, list):
            raw = [raw]
        if not isinstance(events, list):
            events = [events]
        if len(raw) != len(events):
            raise ValueError('raw and events must match')
        if isinstance(event_id, dict):
            event_keys = dict()
            my_event_id = event_id.values()
            for k, v in event_id.items():
                if v not in np.concatenate(events)[:, 1]:
                    warnings.warn('Did not find event id %i' % v,
                                  RuntimeWarning)
                event_keys[v] = k
        elif np.isscalar(event_id):
            my_event_id = [event_id]

        assert len(raw) > 0
        # figure out parameters to use
        idx_offsets = raw[0].time_as_index([self.tmin, self.tmax])
        n_samples = idx_offsets[1] - idx_offsets[0]
        self._n_times = n_samples
        self.times = np.linspace(self.tmin, self.tmax, self._n_times)
        self.info = dict(sfreq=raw[0].info['sfreq'],
                         data_cols=raw[0].info['data_cols'])
        for r in raw[1:]:
            if r.info['sfreq'] != raw[0].info['sfreq']:
                raise RuntimeError('incompatible raw files')
        # process each raw file
        outs = [self._process_raw_events(rr, ee, my_event_id,
                                         event_keys, idx_offsets)
                for rr, ee in zip(raw, events)]

        _samples, _discretes, _events = zip(*outs)
        offset = np.cumsum(np.concatenate(([0], [r.n_samples for r in raw])))
        for ev, off in zip(_events, offset[:-1]):
            ev[:, 0] += off
        _events = np.concatenate(_events)
        self._n_epochs = len(_events)

        # Need to add offsets to our epoch indices
        offset = 0
        for _samp in _samples:
            use_offset = offset
            for _s in _samp:
                _s.epoch_idx += use_offset
                offset += len(_s.epoch_idx.unique())

        # flattening is important, otherwise concatenation fails,
        # the zip returns a somewhat nested structure ...
        _flatten = lambda x: [ii for i in x for ii in i]
        _samples = _flatten(_samples)
        _discretes = _flatten(_discretes)

        # ignore index to allow for sorting + keep unique values
        _data = pd.concat(_samples, ignore_index=True)
        # important for multiple conditions
        _data = _data.sort(['epoch_idx', 'time'])
        self._data = _data
        assert len(_data) == self._n_epochs * len(self.times)
        self._data['times'] = np.tile(self.times, self._n_epochs)
        self._data.set_index(['epoch_idx', 'times'], drop=True,
                             inplace=True, verify_integrity=True)
        assert self._n_epochs == self._data.index.values.max()[0] + 1
        self.info['discretes'] = _discretes
        self.events = _events

    def _process_raw_events(self, raw, events, my_event_id, event_keys,
                            idx_offsets):
        data, times = raw[:]
        discrete_inds = [[] for _ in range(3)]
        sample_inds = dict((k, []) for k in my_event_id)
        saccade_inds, fixation_inds, blink_inds = discrete_inds
        keep_idx = []
        # prevent the evil
        events = events[events[:, 0].argsort()]
        for ii, (event, this_id) in enumerate(events):
            if this_id not in my_event_id:
                continue
            this_time = times[event]
            this_tmin, this_tmax = this_time + self.tmin, this_time + self.tmax
            inds_min, inds_max = raw.time_as_index(this_time)[0] + idx_offsets
            if max([inds_min, inds_max]) >= len(raw.samples):
                continue
            if min([inds_min, inds_max]) < 0:
                continue
            inds = np.arange(inds_min, inds_max)

            sample_inds[this_id].append([inds, ii])
            for kind, parsed in zip(raw.info['event_types'], discrete_inds):
                df = raw.discrete.get(kind, kind)
                assert(set([a <= b for a, b in
                       df[['stime', 'etime']].values]) == set([True]))
                event_in_window = np.where((df['stime'] >= this_tmin) &
                                           (df['etime'] <= this_tmax))
                parsed.append([event_in_window[0], ii, this_id, this_time])
            keep_idx.append(ii)
        events = events[keep_idx]

        discretes = []
        for kind, parsed in zip(['saccades', 'fixations', 'blinks'],
                                discrete_inds):
            this_in = raw.discrete.get(kind, None)
            if this_in is not None:
                this_discrete = Discrete()
                for inds, epochs_idx, this_id, this_time in parsed:
                    this_id = (this_id if event_keys is None else
                               event_keys[this_id])
                    if inds.any().any():
                        df = this_in.ix[inds]
                        df['event_id'] = this_id
                        # don't use -= here b/c pandas complains
                        df['stime'] -= this_time
                        df['etime'] -= this_time
                        this_discrete.append(df)
                    else:
                        this_discrete.append([])
                this_name = kind + '_'
                setattr(self, this_name, this_discrete)  # XXX FIX
                discretes += [this_name]

        _samples = []
        c = np.concatenate
        track_inds = []
        for this_id, values in sample_inds.items():
            ind, _ = zip(*values)
            ind = [i[:self._n_times] for i in ind]
            df = raw.samples.ix[c(ind)]
            this_id = this_id if event_keys is None else event_keys[this_id]
            df['event_id'] = this_id
            count = c([np.repeat(vv, self._n_times) for _, vv in values])
            df['epoch_idx'] = count
            _samples.append(df)
            track_inds.extend([len(i) for i in ind])

        assert set(track_inds) == set([self._n_times])
        n_keep = sum([len(s.epoch_idx.unique()) for s in _samples])
        assert len(events) == n_keep
        return _samples, discretes, events

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
        epoch = epoch[self.info['data_cols']].values.T
        self._current += 1
        if not return_event_id:
            return epoch
        else:
            return epoch, self.events[self._current - 1][-1]

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    @property
    def data(self):
        out = self._data[self.info['data_cols']].values
        out = out.reshape(len(self.events),
                          len(self.times),
                          len(self.info['data_cols']))
        return np.transpose(out, [0, 2, 1])

    @property
    def len(self):
        return self._n_epochs

    @property
    def data_frame(self):
        return self._data

    @property
    def ch_names(self):
        return [k for k in self.data_frame.columns]

    def __getitem__(self, idx):
        out = self.copy()
        if isinstance(idx, string_types):
            if idx not in self.event_id:
                raise ValueError('ID not found')
            idx = self.event_id[idx]
            idx = np.where(self.events[:, -1] == idx)[0]
        elif (isinstance(idx, list) and isinstance(idx[0],
              string_types)):
            idx_list = []
            for ii in idx:
                ii = self.event_id[ii]
                idx_list.append(np.where(self.events[:, -1] == ii)[0])
            idx = np.concatenate(idx_list)
        elif np.isscalar(idx):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = np.arange(*idx.indices(idx.stop))
        # XXX inquire whether Index.map works across pandas versions (fast)
        idx = np.sort(idx)
        midx = [i for i in out._data.index if i[0] in idx]  # ... slow
        out._data = out._data.ix[midx]
        out.events = out.events[idx]
        out._n_epochs = len(idx)
        for discrete in self.info['discretes']:
            disc = vars(self)[discrete]
            setattr(out, discrete, Discrete(disc[k] for k in idx))
        return out

    def copy(self):
        """Return a copy of Epochs.
        """
        return copy.deepcopy(self)

    def plot(self, epoch_idx=None, picks=None, n_chunks=20,
             title_str='#%003i', show=True, draw_discrete=None,
             discrete_colors=None, block=False):
        """ Visualize single trials using Trellis plot.

        Parameters
        ----------
        epoch_idx : array-like | int | None
            The epochs to visualize. If None, the first 20 epochs are shown.
            Defaults to None.
        n_chunks : int
            The number of chunks to use for display.
        picks : array-like | None
            Channels to be included. If None only good data channels are used.
            Defaults to None
        lines : array-like | list of tuple
            Events to draw as vertical lines
        title_str : None | str
            The string formatting to use for axes titles. If None, no titles
            will be shown. Defaults expand to ``#001, #002, ...``
        show : bool
            Whether to show the figure or not.
        draw_discrete : {saccades, blinks, fixations} | list-like | None |
            The events to draw as vertical lines.
        discrete_colors: : list-like | None
            list of str or color objects with length of discrete events drawn.
        block : bool
            Whether to halt program execution until the figure is closed.
            Useful for rejecting bad trials on the fly by clicking on a
            sub plot.

        Returns
        -------
        fig : Instance of matplotlib.figure.Figure
            The figure.
        """
        return plot_epochs(epochs=self, epoch_idx=epoch_idx, picks=picks,
                           n_chunks=n_chunks, title_str=title_str,
                           show=show, draw_discrete=draw_discrete,
                           discrete_colors=discrete_colors,
                           block=block)
