# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from os import path as op
from copy import deepcopy

from ._event import find_events
from ._fixes import string_types
from .viz import plot_calibration, plot_heatmap_raw, plot_raw


class _BaseRaw(object):
    """Base class for Raw"""
    def __init__(self):
        assert self._samples.shape[0] == len(self.info['sample_fields'])
        assert self.times[0] == 0.0
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

    def save(self, fname, overwrite=False):
        """Save data to HD5 format

        Parameters
        ----------
        fname : str
            Filename to use.
        overwrite : bool
            If True, overwrite file (if it exists).
        """
        if op.isfile(fname) and not overwrite:
            raise IOError('file "%s" exists, use overwrite=True to overwrite'
                          % fname)
        try:
            import h5py
        except Exception:
            raise ImportError('h5py could not be imported')
        with h5py.File(fname, mode='w') as fid:
            # samples
            comp_kw = dict(compression='gzip', compression_opts=5)
            s = np.core.records.fromarrays(self._samples)
            s.dtype.names = self.info['sample_fields']
            fid.create_dataset('samples', data=s, **comp_kw)
            # times
            fid.create_dataset('times', data=self._times, **comp_kw)
            # discrete
            dg = fid.create_group('discrete')
            for key, val in self.discrete.items():
                dg.create_dataset(key, data=val, **comp_kw)
            # info (harder)
            info = deepcopy(self.info)
            info['meas_date'] = info['meas_date'].isoformat()
            items = [('eye', '|S256'),
                     ('camera', '|S256'),
                     ('camera_config', '|S256'),
                     ('meas_date', '|S32'),
                     ('ps_units', '|S16'),
                     ('screen_coords', 'f8', self.info['screen_coords'].shape),
                     ('serial', '|S256'),
                     ('sfreq', 'f8'),
                     ('version', '|S256'),
                     ]
            data = np.array([tuple([info[t[0]] for t in items])], dtype=items)
            fid.create_dataset('info', data=data, **comp_kw)
            # calibrations
            cg = fid.create_group('calibrations')
            for ci, cal in enumerate(self.info['calibrations']):
                cg.create_dataset('c%s' % ci, data=cal)

    @property
    def n_samples(self):
        """Number of time samples"""
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

    def plot(self, events=None, title='Raw', show=True):
        """Visualize calibration

        Parameters
        ----------
        events : array | None
            Events associated with the Raw instance.
        title : str
            The title to be displayed.
        show : bool
            Whether to show the figure or not.

        Returns
        -------
        fig : matplotlib.figure.Figure instance
            The resulting figure object.
        """
        return plot_raw(raw=self, events=events, title=title, show=show)

    def plot_heatmap(self, start=None, stop=None, cmap=None, title=None,
                     vmax=None, kernel=dict(size=100, half_width=50),
                     colorbar=True, show=True):
        """ Plot heatmap of X/Y positions on canvas, e.g., screen

        Parameters
        ----------
        start : float | None
            Start time in seconds.
        stop : float | None
            End time in seconds.
        cmap : matplotlib Colormap
            The colormap to use.
        title : str
            The title to be displayed.
        vmax : float | None
            The maximum (and -minimum) value to use for the colormap.
        kernel : dict
            Parameters for the smoothing kernel (size, half_width).
        colorbar : bool
            Whether to show the colorbar.
        show : bool
            Whether to show the figure or not.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The resulting figure object
        """
        return plot_heatmap_raw(raw=self, start=start, stop=stop, cmap=cmap,
                                title=title, vmax=vmax, kernel=kernel,
                                colorbar=colorbar, show=show)

    @property
    def times(self):
        """Time values"""
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
            ps_vals = self['ps', sidx:eidx][0]
            if interp is None:
                fix = np.nan
            elif interp == 'zoh':
                fix = ps_vals[0]
            elif interp == 'linear':
                len_ = eidx - sidx
                fix = np.linspace(ps_vals[0], ps_vals[-1], len_)
            vals = self[:, sidx:eidx][0]
            vals[:] = np.nan
            ps_vals[:] = fix


def read_raw(fname):
    """General Eye-tracker Reader

    Parameters
    ----------
    fname : str
        The name of the eye-tracker data file.
        Files currently supported are EDF and HD5
    """    
    _, ext = op.splitext(fname)
    if ext == '.edf':
        from .edf._raw import RawEDF
        raw = RawEDF(fname)
    elif ext == '.hd5':
        from .hd5._raw import RawHD5
        raw = RawHD5(fname)
    return raw
