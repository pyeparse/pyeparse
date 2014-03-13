# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
from datetime import datetime
try:
    from cStringIO import StringIO as sio
except ImportError:  # py3 has renamed this
    from io import StringIO as sio  # noqa
from os import path as op

from .constants import EDF
from .event import find_events
from .utils import raw_open
from ._py23 import next, string_types
from .viz import plot_calibration, plot_heatmap_raw


def _extract_sys_info(line):
    """Aux function for preprocessing sys info lines"""
    return line[line.find(':'):].strip(': \r\n')


def _get_fields(line):
    line = line.strip().split()
    if line[0] not in ['SAMPLES', 'EVENTS']:
        raise RuntimeError('Unknown type "%s" not EVENTS or SAMPLES' % line[0])
    assert line[1] == 'GAZE'
    eye = line[2]
    assert eye in ['LEFT', 'RIGHT']  # MIGHT NOT WORK FOR BINOCULAR?
    eye = eye[0]
    # always recorded
    sfreq = None
    track = None
    filt = None
    fields = list()
    for fi, f in enumerate(line[3:]):
        if f == 'RATE':
            sfreq = float(line[fi + 4].replace(',', '.'))
        elif f == 'TRACKING':
            track = line[fi + 4]
        elif f == 'FILTER':
            filt = line[fi + 4]
        elif f == 'VEL':
            fields.extend(['xv', 'yv'])
        elif f == 'RES':
            fields.extend(['xres', 'yres'])
        elif f == 'INPUT':
            fields.append('input')

    if any([x is None for x in [sfreq, track, filt]]):
        raise RuntimeError('bad line definition: "%s"' % line)
    return fields, eye, sfreq, track, filt


def _parse_pramble(info, preamble_lines):
    for line in preamble_lines:
        if '!MODE'in line:
            line = line.split()
            info['eye'] = line[-1]
            info['sfreq'] = float(line[-4])
        elif 'DATE:' in line:
            line = _extract_sys_info(line).strip()
            fmt = '%a %b  %d %H:%M:%S %Y'
            info['meas_date'] = datetime.strptime(line, fmt)
        elif 'VERSION:' in line:
            info['version'] = _extract_sys_info(line)
        elif 'CAMERA:' in line:
            info['camera'] = _extract_sys_info(line)
        elif 'SERIAL NUMBER:' in line:
            info['serial'] = _extract_sys_info(line)
        elif 'CAMERA_CONFIG:' in line:
            info['camera_config'] = _extract_sys_info(line)
    return info


def _parse_def_lines(info, def_lines):
    format_lines = list()
    for line in def_lines:
        if line[:7] == 'SAMPLES' or line[:6] == 'EVENTS':
            format_lines.append(line)
        elif EDF.CODE_PUPIL in line:
            if EDF.CODE_PUPIL_AREA in line:
                info['ps_units'] = 'area'
            elif EDF.CODE_PUPIL_DIAMETER in line:
                info['ps_units'] = 'diameter'
        elif 'PRESCALER' in line:
            k, v = line.split()
            info[k] = int(v)
    _parse_put_event_format(info, format_lines)
    return info


def _parse_calibration(info, calib_lines):
    """Parse EL header information"""
    validations = list()
    lines = iter(calib_lines)
    keys = ['point-x', 'point-y', 'offset', 'diff-x', 'diff-y']
    for line in lines:
        if '!CAL VALIDATION ' in line and not 'ABORTED' in line:
            cal_kind = line.split('!CAL VALIDATION ')[1].split()[0]
            n_points = int([c for c in cal_kind if c.isdigit()][0])
            this_validation = []
            while n_points != 0:
                n_points -= 1
                subline = next(lines).split()
                xy = subline[-6].split(',')
                xy_diff = subline[-2].split(',')
                vals = xy[:2] + [subline[-4]] + xy_diff[:2]
                assert len(vals) == 5
                vals = [float(v) for v in vals]
                this_validation.append(vals)
            this_validation = np.array(this_validation).T
            validations.append(this_validation)
        elif any([k in line for k in EDF.MLINES]):
            additional_lines = []
            # additional lines on our way to the
            # empty line block tail
            while True:
                this_line = next(lines)
                if not this_line.strip('\n') or 'MSG' in line:
                    break
                additional_lines.append(this_line)
            line += '  '
            line += '; '.join(additional_lines)
        elif 'DISPLAY_COORDS' in line:
            info['screen_coords'] = np.array(line.split()[-2:],
                                             dtype='i8')
    if len(validations) > 0:
        out = dict()
        for key, val in zip(keys, validations[-1]):
            out[key] = val
    else:
        out = validations
    return out


def _parse_put_event_format(info, def_lines):
    """Figure out what all our fields are from SAMPLES & EVENTS lines"""
    if len(def_lines) > 2:
        def_lines = def_lines[:2]

    for l,  k in zip(sorted(def_lines), ['EVENTS', 'SAMPLES']):
        l.startswith(k)

    saccade_fields = ['eye', 'stime', 'etime', 'dur', 'sxp', 'syp',
                      'exp', 'eyp', 'ampl', 'pv']
    fixation_fields = ['eye', 'stime', 'etime', 'dur', 'axp', 'ayp', 'aps']

    for line in def_lines:
        extra, eye, sfreq, track, filt = _get_fields(line)
        if line.startswith('SAMPLES'):
            fields = ['time', 'xpos', 'ypos', 'ps']
            key = 'sample_fields'
        else:
            fields = ['eye', 'stime', 'etime', 'dur', 'xpos', 'ypos', 'ps']
            key = 'event_fields'
            saccade_fields.extend(extra)
            fixation_fields.extend(extra)

        fields.extend(extra)
        fields.append('status')
        for k, v in zip(['sfreq', 'track', 'filt', 'eye'],
                        [sfreq, track, filt, eye]):
            if k in info:
                assert info[k] == v
            else:
                info[k] = v
        info[key] = fields
    info['saccade_fields'] = saccade_fields
    info['fixation_fields'] = fixation_fields
    info['blink_fields'] = EDF.BLINK_FIELDS
    info['discretes'] = []


def _read_samples_string(samples):
    """Triage sample interpretation using Pandas or Numpy"""
    try:
        import pandas as pd
    except:
        samples = np.genfromtxt(samples, dtype=np.float64)
    else:
        samples = pd.read_table(samples, sep='[ \t]+',
                                na_values=['.', '...']).values.T.copy()
    return samples


def _read_raw(fname):
        def_lines, esacc, efix, eblink, calibs, preamble, messages = \
            [list() for _ in range(7)]
        samples = []
        started = False
        event_types = ['saccades', 'fixations', 'blinks']
        runs = []
        if not op.isfile(fname):
            raise IOError('file "%s" not found' % fname)
        with raw_open(fname) as fid:
            for line in fid:
                if line[0] in ['#/;']:  # comment line, ignore it
                    continue
                if not started:
                    if line.startswith('**'):
                        preamble.append(line)
                    else:
                        started = True
                else:
                    if line[0].isdigit():
                        samples.append(line)
                    elif line.startswith(EDF.CODE_ESAC):
                        esacc.append(line)
                    elif line.startswith(EDF.CODE_EFIX):
                        efix.append(line)
                    elif line.startswith(EDF.CODE_EBLINK):
                        eblink.append(line)
                    elif line.startswith('MSG'):
                        messages.append(line)
                    elif line.startswith('>') and 'CALIBRATION' in line:
                        # Add another calibration section, set split for
                        # raw parser
                        if samples:
                            runs.append(dict(def_lines=def_lines,
                                             samples=samples,
                                             esacc=esacc, efix=efix,
                                             eblink=eblink,
                                             calibs=calibs, preamble=preamble,
                                             messages=messages))
                            def_lines, esacc, efix, eblink, calibs, preamble, \
                                messages = [list() for _ in range(7)]
                            samples = []
                        calib_lines = []
                        while True:
                            subline = next(fid)
                            if subline.startswith('START'):
                                break
                            calib_lines.append(subline)
                        calibs.extend(calib_lines)
                    elif any([line.startswith(k) for k in
                             EDF.DEF_LINES]):
                        def_lines.append(line)
                    elif EDF.CODE_SSAC == line[:len(EDF.CODE_SSAC)]:
                        pass
                    elif EDF.CODE_SFIX == line[:len(EDF.CODE_SFIX)]:
                        pass
                    elif EDF.CODE_SBLINK == line[:len(EDF.CODE_SBLINK)]:
                        pass
                    elif line.startswith('END'):
                        pass
                    elif line.startswith('INPUT'):
                        pass
                    else:
                        raise RuntimeError('data not understood: "%s"'
                                           % line)
        runs.append(dict(def_lines=def_lines, samples=samples,
                         esacc=esacc, efix=efix, eblink=eblink,
                         calibs=calibs, preamble=preamble,
                         messages=messages))

        for keys, values in zip(zip(*[d.keys() for d in runs]),
                                zip(*[d.values() for d in runs])):
            if 'def_lines' not in keys:
                assert not any([k == values[0] for k in values[1:]])
            else:
                assert all([k == values[0] for k in values[1:]])
        # parse the header
        samples_runs, discrete_runs = [], []
        discrete = dict()
        fs = None
        info = None
        _t_zero = None
        offset = 0
        assert len(runs) > 0
        for ii, run in enumerate(runs):
            this_info = dict(calibration=list())
            if run['preamble']:
                _parse_pramble(this_info, run['preamble'])

            _parse_def_lines(this_info, run['def_lines'])
            if not all([x in this_info
                        for x in ['sample_fields', 'event_fields']]):
                raise RuntimeError('could not parse header')
            if run['calibs']:
                validation = _parse_calibration(this_info, run['calibs'])
                this_info['calibration'] = [validation]
                if 'sample_fields' not in this_info:
                    _parse_put_event_format(this_info, run['def_lines'])
            if info is None:
                info = this_info
                info['event_types'] = event_types
            else:
                info['calibration'] += this_info['calibration']
            fs = info['sfreq']
            assert fs == this_info['sfreq']
            samples = run['samples']
            samples = sio(''.join(samples))
            samples = _read_samples_string(samples)
            assert len(np.unique(samples[0])) == samples.shape[1]
            this_t_zero = samples[0, 0]
            if _t_zero is None:
                _t_zero = this_t_zero
            samples[0] -= this_t_zero
            samples[0] /= 1e3
            samples += offset
            samples_runs.append(samples)

            # parse discretes
            this_discrete = {}
            kind_list = [run['esacc'], run['efix'], run['eblink']]
            for s, kind in zip(event_types, kind_list):
                d = np.genfromtxt(sio(''.join(kind)), dtype=None)
                disc = dict()
                for ii, key in enumerate(info[s[:-1] + '_fields']):
                    disc[key] = d['f%s' % (ii + 1)]  # first field is junk
                for key in ('stime', 'etime'):
                    disc[key] = (disc[key] - this_t_zero) / 1e3 + offset
                assert np.all(disc['stime'] < disc['etime'])
                this_discrete[s] = disc

            # parse messages
            times = list()
            msgs = list()
            for message in run['messages']:
                x = message.strip().split(None, 2)
                assert x[0] == 'MSG'
                times.append(x[1])
                msgs.append(x[2])
            times = (np.array(times, np.float64) - this_t_zero) / 1e3 + offset
            msgs = np.array(msgs, dtype='O')
            this_discrete['messages'] = dict(time=times, msg=msgs)
            discrete_runs.append(this_discrete)

            # set offset for next group
            offset += samples[0, -1] + 1. / fs

        # combine all fields
        for kind in (event_types + ['messages']):
            discrete[kind] = dict()
            for col in discrete_runs[0][kind].keys():
                concat = np.concatenate([d[kind][col]
                                         for d in discrete_runs])
                discrete[kind][col] = concat
        samples = np.concatenate(samples_runs, axis=1)
        return info, discrete, samples, _t_zero


class Raw(object):
    """ Represent EyeLink 1000 ASCII files in Python

    Parameters
    ----------
    fname : str
        The name of the ASCII converted EDF file.
    """
    def __init__(self, fname):
        info, discrete, samples, t_zero = _read_raw(fname)
        self.info = info
        self.discrete = discrete
        self._samples = samples
        self._t_zero = t_zero
        assert self._samples.shape[0] == len(self.info['sample_fields'])

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
            if idx[0] not in self.info['sample_fields']:
                raise KeyError('string idx "%s" must be one of %s'
                               % (idx, self.info['sample_fields']))
            idx = list(idx)
            idx[0] = self.info['sample_fields'].index(idx[0])
            idx = tuple(idx)
        if len(idx) > 2:
            raise ValueError('indices must have at most two elements')
        elif len(idx) == 1:
            idx = (idx[0], slice(None))
        data = self._samples[idx]
        times = self._samples[0][idx[1:]]
        return data, times

    def _di(self, key):
        """Helper to get the sample dict index"""
        if key not in self.info['sample_fields']:
            raise KeyError('key "%s" not in sample fields %s'
                           % (key, self.info['sample_fields']))
        return self.info['sample_fields'].index(key)

    @property
    def n_samples(self):
        return self._samples.shape[1]

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
        return self._samples[0].copy()

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
