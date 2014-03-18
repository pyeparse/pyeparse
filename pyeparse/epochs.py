# -*- coding: utf-8 -*-
# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from copy import deepcopy
import numpy as np
from scipy.optimize import fmin_slsqp
import warnings

from ._event import Discrete
from .viz import plot_epochs
from .utils import discrete_types, pupil_kernel
from ._py23 import string_types
from .parallel import parallel_func


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
    ignore_missing : bool
        If True, do not warn if no events were found.

    Returns
    -------
    epochs : instance of Epochs
        The epoched dataset.
    """
    def __init__(self, raw, events, event_id, tmin, tmax,
                 ignore_missing=False):
        if np.isscalar(event_id) and not isinstance(event_id, string_types):
            if event_id != int(event_id):
                raise RuntimeError('event_id must be an integer')
            event_id = int(event_id)
            event_id = {str(event_id): event_id}
        if not isinstance(event_id, dict):
            raise RuntimeError('event_id must be an int or dict')
        self.event_id = deepcopy(event_id)
        self.tmin = tmin
        self.tmax = tmax
        self._current = 0
        if not isinstance(raw, list):
            raw = [raw]
        if not isinstance(events, list):
            events = [events]
        if len(raw) != len(events):
            raise ValueError('raw and events must match')
        event_keys = dict()
        my_event_id = event_id.values()
        for k, v in event_id.items():
            if (not ignore_missing and
                    v not in np.concatenate(events)[:, 1]):
                warnings.warn('Did not find event id %i' % v,
                              RuntimeWarning)
            event_keys[v] = k

        assert len(raw) > 0
        # figure out parameters to use
        idx_offsets = raw[0].time_as_index([self.tmin, self.tmax])
        n_times = idx_offsets[1] - idx_offsets[0]
        self._n_times = n_times
        self.info = dict(sfreq=raw[0].info['sfreq'],
                         data_cols=deepcopy(raw[0].info['sample_fields'][1:]),
                         ps_units=raw[0].info['ps_units'])
        for r in raw[1:]:
            if r.info['sfreq'] != raw[0].info['sfreq']:
                raise RuntimeError('incompatible raw files')
        # process each raw file
        outs = [self._process_raw_events(rr, ee, my_event_id,
                                         event_keys, idx_offsets)
                for rr, ee in zip(raw, events)]

        _samples, _discretes, _events = zip(*outs)
        t_off = np.cumsum(np.concatenate(([0], [r.n_samples for r in raw])))
        t_off = t_off[:-1]
        for ev, off in zip(_events, t_off):
            ev[:, 0] += off
        # Calculate offsets for epoch indices
        e_off = np.cumsum(np.concatenate(([0], [len(e) for e in _events])))
        _events = np.concatenate(_events)
        self.events = _events

        self._data = np.empty((e_off[-1], len(self.info['data_cols']),
                               self.n_times))
        # Need to add offsets to our epoch indices
        for _samp, e1, e2, t in zip(_samples, e_off[:-1], e_off[1:], t_off):
            for ii in range(len(_samp)):
                self._data[e1:e2, ii] = _samp[ii]

        # deal with discretes
        for kind in discrete_types:
            this_discrete = Discrete()
            for d in _discretes:
                this_discrete.extend(d[kind])
            assert len(this_discrete) == len(self.events)
            setattr(self, kind, this_discrete)
        self.info['discretes'] = discrete_types

    def _process_raw_events(self, raw, events, my_event_id, event_keys,
                            idx_offsets):
        times = raw.times.copy()
        sample_inds = []
        keep_idx = []
        # prevent the evil
        events = events[events[:, 0].argsort()]
        discretes = dict()
        for kind in raw.info['event_types']:
            discretes[kind] = Discrete()
        for ii, (event, this_id) in enumerate(events):
            if this_id not in my_event_id:
                continue
            this_time = times[event]
            this_tmin, this_tmax = this_time + self.tmin, this_time + self.tmax
            inds_min, inds_max = raw.time_as_index(this_time)[0] + idx_offsets
            if max([inds_min, inds_max]) >= len(raw):
                continue
            if min([inds_min, inds_max]) < 0:
                continue
            inds = np.arange(inds_min, inds_max)
            sample_inds.append(inds)
            keep_idx.append(ii)

            for kind in raw.info['event_types']:
                this_disc = discretes[kind]
                df = raw.discrete[kind]
                comp_1 = df['stime']
                comp_2 = df['etime'] if 'etime' in df else df['stime']
                idx = np.where((comp_1 >= this_tmin) & (comp_2 <= this_tmax))
                for ii in idx:
                    subdict = dict()
                    for key in df.keys():
                        subdict[key] = df[key][idx].copy()
                    subdict['stime'] -= this_time
                    if 'etime' in subdict:
                        subdict['etime'] -= this_time
                    this_disc.append(subdict)
        events = events[keep_idx]
        sample_inds = np.array(sample_inds)
        samples = raw[1:, sample_inds][0]
        for kind in raw.info['event_types']:
            assert len(discretes[kind]) == len(events)
        return samples, discretes, events

    def __len__(self):
        return len(self.events)

    def __repr__(self):
        s = '<Epochs | {0} events | tmin: {1} tmax: {2}>'
        return s.format(len(self), self.tmin, self.tmax)

    def __iter__(self):
        """To make iteration over epochs easy.
        """
        self._current = 0
        return self

    def next(self, return_event_id=False):
        """To make iteration over epochs easy.
        """
        if self._current >= len(self):
            raise StopIteration
        epoch = self._data[self._current]
        event = self.events[self._current, -1]
        self._current += 1
        if not return_event_id:
            out = epoch
        else:
            out = (epoch, event)
        return out

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    @property
    def data(self):
        return self._data

    def get_data(self, kind):
        """Get data of a particular kind

        Parameters
        ----------
        kind : str
            Kind of data go obtain. Must be one of ``self.info['data_cols']``.

        Returns
        -------
        data : array
            Array of data, n_epochs x n_times.
        """
        if kind not in self.info['data_cols']:
            raise ValueError('kind "%s" must be one of %s'
                             % (kind, self.info['data_cols']))
        return self._data[:, self.info['data_cols'].index(kind)]

    @property
    def data_frame(self):
        raise NotImplementedError

    @property
    def ch_names(self):
        return self.info['data_cols']

    @property
    def n_times(self):
        return self._n_times

    @property
    def times(self):
        return (np.arange(self.n_times).astype(float) / self.info['sfreq']
                + self.tmin)

    def _str_to_idx(self, string):
        """Convert epoch string label to set of indices"""
        if string not in self.event_id:
            raise IndexError('ID "%s" not found, must be one of %s'
                             % (string, list(self.event_id.keys())))
        idx = np.where(self.events[:, -1] == self.event_id[string])[0]
        return idx

    def __getitem__(self, idx):
        out = self.copy()
        if isinstance(idx, string_types):
            idx = self._str_to_idx(idx)
        elif isinstance(idx, list):
            if all([isinstance(ii, string_types) for ii in idx]):
                idx = np.concatenate([self._str_to_idx(ii) for ii in idx])
        if isinstance(idx, slice):
            idx = np.arange(len(self))[idx]
        idx = np.atleast_1d(np.sort(idx))
        out._data = out._data[idx]
        out.events = out.events[idx]
        for discrete in self.info['discretes']:
            disc = getattr(self, discrete)
            setattr(out, discrete, Discrete(disc[k] for k in idx))
        return out

    def time_as_index(self, times):
        """Convert time to indices

        Parameters
        ----------
        times : list-like | float | int
            List of numbers or a number representing points in time.

        Returns
        -------
        index : ndarray
            Indices corresponding to the times supplied.
        """
        index = (np.atleast_1d(times) - self.times[0]) * self.info['sfreq']
        return index.astype(int)

    def copy(self):
        """Return a copy of Epochs.
        """
        return deepcopy(self)

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

    def combine_event_ids(self, old_event_ids, new_event_id):
        """Collapse event_ids into a new event_id

        Parameters
        ----------
        old_event_ids : str, or list
            Conditions to collapse together.
        new_event_id : dict, or int
            A one-element dict (or a single integer) for the new
            condition. Note that for safety, this cannot be any
            existing id (in epochs.event_id.values()).

        Notes
        -----
        This For example (if epochs.event_id was {'Left': 1, 'Right': 2}:

            combine_event_ids(epochs, ['Left', 'Right'], {'Directional': 12})

        would create a 'Directional' entry in epochs.event_id replacing
        'Left' and 'Right' (combining their trials).
        """
        old_event_ids = np.asanyarray(old_event_ids)
        if isinstance(new_event_id, int):
            new_event_id = {str(new_event_id): new_event_id}
        else:
            if not isinstance(new_event_id, dict):
                raise ValueError('new_event_id must be a dict or int')
            if not len(list(new_event_id.keys())) == 1:
                raise ValueError('new_event_id dict must have one entry')
        new_event_num = list(new_event_id.values())[0]
        if not isinstance(new_event_num, int):
            raise ValueError('new_event_id value must be an integer')
        if new_event_num in self.event_id.values():
            raise ValueError('new_event_id value must not already exist')
        old_event_nums = np.array([self.event_id[key]
                                   for key in old_event_ids])
        # find the ones to replace
        inds = np.any(self.events[:, 1][:, np.newaxis] ==
                      old_event_nums[np.newaxis, :], axis=1)
        # replace the event numbers in the events list
        self.events[inds, 1] = new_event_num
        # delete old entries
        for key in old_event_ids:
            self.event_id.pop(key)
        # add the new entry
        self.event_id.update(new_event_id)

    def _key_match(self, key):
        """Helper function for event dict use"""
        if key not in self.event_id:
            raise KeyError('Event "%s" is not in Epochs.' % key)
        return self.events[:, 1] == self.event_id[key]

    def drop_epochs(self, indices):
        """Drop epochs based on indices or boolean mask

        Parameters
        ----------
        indices : array of ints or bools
            Set epochs to remove by specifying indices to remove or a boolean
            mask to apply (where True values get removed). Events are
            correspondingly modified.
        """
        indices = np.atleast_1d(indices)

        if indices.ndim > 1:
            raise ValueError("indices must be a scalar or a 1-d array")

        if indices.dtype == bool:
            indices = np.where(indices)[0]

        out_of_bounds = (indices < 0) | (indices >= len(self.events))
        if out_of_bounds.any():
            raise IndexError("Epoch index %d is out of bounds"
                             % indices[out_of_bounds][0])

        old_idx = np.delete(np.arange(len(self)), indices)
        self.events = np.delete(self.events, indices, axis=0)
        self._data = np.delete(self._data, indices, axis=0)
        for key in self.info['discretes']:
            val = getattr(self, key)
            setattr(self, key, [val[ii] for ii in old_idx])

    def equalize_event_counts(self, event_ids, method='mintime'):
        """Equalize the number of trials in each condition

        Parameters
        ----------
        event_ids : list
            The event types to equalize. Each entry in the list can either be
            a str (single event) or a list of str. In the case where one of
            the entries is a list of str, event_ids in that list will be
            grouped together before equalizing trial counts across conditions.
        method : str
            If 'truncate', events will be truncated from the end of each event
            list. If 'mintime', timing differences between each event list will
            be minimized.

        Returns
        -------
        epochs : instance of Epochs
            The modified Epochs instance.
        indices : array of int
            Indices from the original events list that were dropped.

        Notes
        ----
        This method operates in-place.
        """
        epochs = self
        if len(event_ids) == 0:
            raise ValueError('event_ids must have at least one element')
        # figure out how to equalize
        eq_inds = list()
        for eq in event_ids:
            eq = np.atleast_1d(eq)
            # eq is now a list of types
            key_match = np.zeros(epochs.events.shape[0])
            for key in eq:
                key_match = np.logical_or(key_match, epochs._key_match(key))
            eq_inds.append(np.where(key_match)[0])

        event_times = [epochs.events[eq, 0] for eq in eq_inds]
        indices = _get_drop_indices(event_times, method)
        # need to re-index indices
        indices = np.concatenate([eq[inds]
                                  for eq, inds in zip(eq_inds, indices)])
        epochs.drop_epochs(indices)
        # actually remove the indices
        return epochs, indices

    def pupil_zscores(self, baseline=(None, 0)):
        """Get normalized pupil data

        Parameters
        ----------
        baseline : list
            2-element list of time points to use as baseline.
            The default is (None, 0), which uses all negative time.

        Returns
        -------
        pupil_data : array
            An n_epochs x n_time array of pupil size data.
        """
        if 'ps' not in self.info['data_cols']:
            raise RuntimeError('no pupil data')
        if len(baseline) != 2:
            raise RuntimeError('baseline must be a 2-element list')
        baseline = np.array(baseline)
        if baseline[0] is None:
            baseline[0] = self.times[0]
        if baseline[1] is None:
            baseline[1] = self.times[-1]
        baseline = self.time_as_index(baseline)
        zs = self.get_data('ps')
        std = np.nanstd(zs.flat)
        bl = np.nanmean(zs[:, baseline[0]:baseline[1] + 1], axis=1)
        zs -= bl[:, np.newaxis]
        zs /= std
        return zs

    def deconvolve(self, spacing=0.1, baseline=(None, 0), bounds=None,
                   max_iter=500, n_jobs=1):
        """Deconvolve pupillary responses

        Parameters
        ----------
        spacing : float | array
            Spacing of time points to use for deconvolution. Can also
            be an array to directly specify time points to use.
        baseline : list
            2-element list of time points to use as baseline.
            The default is (None, 0), which uses all negative time.
            This is passed to pupil_zscores().
        bounds : 2-element array | None
            Limits for deconvolution values. Can be, e.g. (0, np.inf) to
            constrain to positive values.
        max_iter : int
            Maximum number of iterations of minimization algorithm.
        n_jobs : array
            Number of jobs to run in parallel.

        Returns
        -------
        fit : array
            Array of fits, of size n_epochs x n_fit_times.
        times : array
            The array of times at which points were fit.

        Notes
        -----
        This method is adapted from:

            Wierda et al., 2012, "Pupil dilation deconvolution reveals the
            dynamics of attention at high temporal resolution."

        See: http://www.pnas.org/content/109/22/8456.long

        Our implementation does not, by default, force all weights to be
        greater than zero. It also does not do first-order detrending,
        which the Wierda paper discusses implementing.
        """
        if bounds is not None:
            bounds = np.array(bounds)
            if bounds.ndim != 1 or bounds.size != 2:
                raise RuntimeError('bounds must be 2-element array or None')

        # get the data (and make sure it exists)
        pupil_data = self.pupil_zscores(baseline)

        # set up parallel function (and check n_jobs)
        parallel, p_fun, n_jobs = parallel_func(_do_deconv, n_jobs)

        # figure out where the samples go
        n_samp = self.n_times
        if not isinstance(spacing, (np.ndarray, tuple, list)):
            times = np.arange(self.times[0], self.times[-1], spacing)
            times = np.unique(times)
        else:
            times = np.asanyarray(spacing)
        samples = self.time_as_index(times)
        if len(samples) == 0:
            warnings.warn('No usable samples')
            return np.array([]), np.array([])

        # convert bounds to slsqp representation
        if bounds is not None:
            bounds = np.array([bounds for _ in range(len(samples))])
        else:
            bounds = []  # compatible with old version of scipy

        # Build the convolution matrix
        kernel = pupil_kernel(self.info['sfreq'])
        conv_mat = np.zeros((n_samp, len(samples)))
        for li, loc in enumerate(samples):
            eidx = min(loc + len(kernel), n_samp)
            conv_mat[loc:eidx, li] = kernel[:eidx-loc]

        # do the fitting
        fit_fails = parallel(p_fun(data, conv_mat, bounds, max_iter)
                             for data in np.array_split(pupil_data, n_jobs))
        fit = np.concatenate([f[0] for f in fit_fails])
        fails = np.concatenate([f[1] for f in fit_fails])
        if np.any(fails):
            reasons = ', '.join(str(r) for r in
                                np.setdiff1d(np.unique(fails), [0]))
            warnings.warn('%i/%i fits did not converge (reasons: %s)'
                          % (np.sum(fails != 0), len(fails), reasons))
        return fit, times


def _do_deconv(pupil_data, conv_mat, bounds, max_iter):
    """Helper to parallelize deconvolution"""
    x0 = np.ones(conv_mat.shape[1])
    fit = np.empty((len(pupil_data), conv_mat.shape[1]))
    failed = np.empty(len(pupil_data))
    for di, data in enumerate(pupil_data):
        out = fmin_slsqp(_score, x0, args=(data, conv_mat), epsilon=1e-3,
                         bounds=bounds, disp=False, full_output=True,
                         iter=max_iter, acc=1e-6)
        fit[di, :] = out[0]
        failed[di] = out[3]
    return fit, failed


def _score(vals, x_0, conv_mat):
    return np.mean((x_0 - conv_mat.dot(vals)) ** 2)


def _get_drop_indices(event_times, method):
    """Helper to get indices to drop from multiple event timing lists"""
    small_idx = np.argmin([e.shape[0] for e in event_times])
    small_e_times = event_times[small_idx]
    if not method in ['mintime', 'truncate']:
        raise ValueError('method must be either mintime or truncate, not '
                         '%s' % method)
    indices = list()
    for e in event_times:
        if method == 'mintime':
            mask = _minimize_time_diff(small_e_times, e)
        else:
            mask = np.ones(e.shape[0], dtype=bool)
            mask[small_e_times.shape[0]:] = False
        indices.append(np.where(np.logical_not(mask))[0])

    return indices


def _minimize_time_diff(t_shorter, t_longer):
    """Find a boolean mask to minimize timing differences"""
    keep = np.ones((len(t_longer)), dtype=bool)
    scores = np.ones((len(t_longer)))
    for iter in range(len(t_longer) - len(t_shorter)):
        scores.fill(np.inf)
        # Check every possible removal to see if it minimizes
        for idx in np.where(keep)[0]:
            keep[idx] = False
            scores[idx] = _area_between_times(t_shorter, t_longer[keep])
            keep[idx] = True
        keep[np.argmin(scores)] = False
    return keep


def _area_between_times(t1, t2):
    """Quantify the difference between two timing sets"""
    x1 = list(range(len(t1)))
    x2 = list(range(len(t2)))
    xs = np.concatenate((x1, x2))
    return np.sum(np.abs(np.interp(xs, x1, t1) - np.interp(xs, x2, t2)))
