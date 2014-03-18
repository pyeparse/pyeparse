# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np

from ._event import find_events
from ._py23 import string_types
from .viz import plot_calibration, plot_heatmap_raw


class _BaseRaw(object):
    """Base class for Raw"""
    def __init__(self):
        assert self._samples.shape[0] == len(self.info['sample_fields'])
        assert self.times[0] == 0
        assert isinstance(self.info['sfreq'], float)
        dt = np.abs(np.diff(self.times) - (1. / self.info['sfreq']))
        assert np.all(dt < 1e-6)

    def __repr__(self):
        return '<Raw | {0} samples>'.format(self.n_samples)

    def __getitem__(self, idx):
        if isinstance(idx, string_types):
            idx = (idx,)
        elif isinstance(idx, slice):
            idx = (idx,)
        if not isinstance(idx, tuple):
            raise TypeError('index must be a string, slice, or tuple')
        if isinstance(idx[0], string_types):
            idx = list(idx)
            idx[0] = self._di(idx[0])
            idx = tuple(idx)
        if len(idx) > 2:
            raise ValueError('indices must have at most two elements')
        elif len(idx) == 1:
            idx = (idx[0], slice(None))
        data = self._samples[idx]
        times = self.times[idx[1:]]
        return data, times

    def _di(self, key):
        """Helper to get the sample dict index"""
        if key not in self.info['sample_fields']:
            raise KeyError('key "%s" not in sample fields %s'
                           % (key, self.info['sample_fields']))
        return self.info['sample_fields'].index(key)

    @property
    def n_samples(self):
        return len(self.times)

    def __len__(self):
        return self.n_samples

    def plot_calibration(self, title='Calibration', show=True):
        """Visualize calibration

        Parameters
        ----------
        title : str
            The title to be displayed.
        show : bool
            Whether to show the figure or not.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The resulting figure object
        """
        return plot_calibration(raw=self, title=title, show=show)

    def plot_heatmap(self, start=None, stop=None, cmap=None,
                     title=None, kernel=dict(size=100, half_width=50),
                     colorbar=None, show=True):
        """ Plot heatmap of X/Y positions on canvas, e.g., screen

        Parameters
        ----------
        start : float | None
            Start time in seconds.
        stop : float | None
            End time in seconds.
        title : str
            The title to be displayed.
        show : bool
            Whether to show the figure or not.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The resulting figure object
        """
        plot_heatmap_raw(raw=self, start=start, stop=stop, cmap=cmap,
                         title=title, kernel=kernel, colorbar=colorbar,
                         show=show)

    @property
    def times(self):
        return self._times

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
        index = np.atleast_1d(times) * self.info['sfreq']
        return index.astype(int)

    def find_events(self, pattern, event_id):
        """Find parsed messages

        Parameters
        ----------
        raw : instance of pyeparse.raw.Raw
            the raw file to find events in.
        pattern : str | callable
            A substring to be matched or a callable that matches
            a string, for example ``lambda x: 'my-message' in x``
        event_id : int
            The event id to use.

        Returns
        -------
        idx : instance of numpy.ndarray (times, event_id)
            The indices found.
        """
        return find_events(raw=self, pattern=pattern, event_id=event_id)

    def remove_blink_artifacts(self, interp='linear', borders=(0.025, 0.1),
                               use_only_blink=False):
        """Remove blink artifacts from gaze data

        This function uses the timing of saccade events to clean up
        pupil size data.

        Parameters
        ----------
        interp : str | None
            If string, can be 'linear' or 'zoh' (zeroth-order hold).
            If None, no interpolation is done, and extra ``nan`` values
            are inserted to help clean data. (The ``nan`` values inserted
            by Eyelink itself typically do not span the entire blink
            duration.)
        borders : float | list of float
            Time on each side of the saccade event to use as a border
            (in seconds). Can be a 2-element list to supply different borders
            for before and after the blink. This will be additional time
            that is eliminated as invalid and interpolated over
            (or turned into ``nan``).
        use_only_blink : bool
            If True, interpolate only over regions where a blink event
            occurred. If False, interpolate over all regions during
            which saccades occurred -- this is generally safer because
            Eyelink will not always categorize blinks correctly.
        """
        if interp is not None and interp not in ['linear', 'zoh']:
            raise ValueError('interp must be None, "linear", or "zoh", not '
                             '"%s"' % interp)
        borders = np.array(borders)
        if borders.size == 1:
            borders == np.array([borders, borders])
        blinks = self.discrete['blinks']['stime']
        starts = self.discrete['saccades']['stime']
        ends = self.discrete['saccades']['etime']
        # only use saccades that enclose a blink
        if use_only_blink:
            use = np.searchsorted(ends, blinks)
            ends = ends[use]
            starts = starts[use]
        starts = starts - borders[0]
        ends = ends + borders[1]
        # eliminate overlaps and unusable ones
        etime = (self.n_samples - 1) / self.info['sfreq']
        use = np.logical_and(starts > 0, ends < etime)
        starts = starts[use]
        ends = ends[use]
        use = starts[1:] > ends[:-1]
        starts = starts[np.concatenate([[True], use])]
        ends = ends[np.concatenate([use, [True]])]
        assert len(starts) == len(ends)
        for stime, etime in zip(starts, ends):
            sidx, eidx = self.time_as_index([stime, etime])
            vals = self['ps', sidx:eidx][0]
            if interp is None:
                fix = np.nan
            elif interp == 'zoh':
                fix = vals[0]
            elif interp == 'linear':
                len_ = eidx - sidx
                fix = np.linspace(vals[0], vals[-1], len_)
            vals[:] = fix
