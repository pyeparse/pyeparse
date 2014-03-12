# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import copy
from datetime import datetime
try:
    from cStringIO import StringIO as sio
except ImportError:  # py3 has renamed this
    from io import StringIO as sio  # noqa
import tempfile
import subprocess
import shutil
from os import path as op

from .constants import EDF
from .event import find_events
from .utils import next, string_types
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
                this_validation.append({'point-x': xy[0],
                                        'point-y': xy[1],
                                        'offset': subline[-4],
                                        'diff-x': xy_diff[0],
                                        'diff-y': xy_diff[1]})
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

    return validations[-1]


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
        if line[:7] == 'SAMPLES':
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
    info['discretes'] = []


def _convert_edf(fname):
    """Helper to convert EDF to ASC on the fly for conversion"""
    # Ideally we will eventually handle the binary files directly
    out_dir = tempfile.mkdtemp('edf2asc')
    out_fname = op.join(out_dir, 'temp.asc')
    p = subprocess.Popen(['edf2asc', fname, out_fname], stderr=subprocess.PIPE,
                         stdout=subprocess.PIPE)
    stdout_, stderr = p.communicate()
    if p.returncode != 255:
        print((p.returncode, stdout_, stderr))
        raise RuntimeError('Could not convert EDF to ASC')
    return out_fname, out_dir


class Raw(object):
    """ Represent EyeLink 1000 ASCII files in Python

    Parameters
    ----------
    fname : str
        The name of the ASCII converted EDF file.
    """
    def __init__(self, fname):

        def_lines, samples, esacc, efix, eblink, calibs, preamble, \
            messages = [list() for _ in range(8)]
        started = False
        runs = []
        if not op.isfile(fname):
            raise IOError('file "%s" not found' % fname)
        del_dir = None
        if fname.endswith('.edf'):
            fname, del_dir = _convert_edf(fname)
        with open(fname, 'r') as fid:
            for line in fid:
                if line[0] in ['#/;']:  # comment line, ignore it
                    continue
                if not started:
                    if line.startswith('**'):
                        preamble.append(line)
                    else:
                        started = True
                        continue  # don't parse empty  line

                if started:
                    if line[0].isdigit():
                        samples.append(line)
                    elif EDF.CODE_ESAC == line[:len(EDF.CODE_ESAC)]:
                        esacc.append(line)
                    elif EDF.CODE_EFIX == line[:len(EDF.CODE_EFIX)]:
                        efix.append(line)
                    elif EDF.CODE_EBLINK in line[:len(EDF.CODE_EBLINK)]:
                        eblink.append(line)
                    elif 'MSG' == line[:3]:
                        messages.append(line)
                    elif 'CALIBRATION' in line and line.startswith('>'):
                        # Add another calibration section, set split for
                        # raw parser
                        if samples:
                            runs.append(dict(def_lines=def_lines,
                                             samples=samples,
                                             esacc=esacc, efix=efix,
                                             eblink=eblink,
                                             calibs=calibs, preamble=preamble,
                                             messages=messages))
                            def_lines, samples, esacc, efix, eblink, calibs, \
                                preamble, \
                                messages = [list() for _ in range(8)]
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
                    elif 'END' == line[:3]:
                        pass
                    elif 'INPUT' == line[:5]:
                        pass
                    else:
                        raise RuntimeError('data not understood: "%s"'
                                           % line)
        if del_dir is not None:
            shutil.rmtree(del_dir)  # to remove temporary conversion files
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
        info_runs, samples_runs, discrete_runs = [], [], []
        for ii, run in enumerate(runs):
            info = {}
            if run['preamble']:
                _parse_pramble(info, run['preamble'])

            _parse_def_lines(info, run['def_lines'])
            if not all([x in info
                        for x in ['sample_fields', 'event_fields']]):
                raise RuntimeError('could not parse header')

            if run['calibs']:
                validation = _parse_calibration(info, run['calibs'])
                validation = np.array(validation)
                info['calibration'] = validation
                if 'sample_fields' not in info:
                    _parse_put_event_format(info, run['def_lines'])
            samples = run['samples']
            samples = ''.join(samples)
            samples = np.genfromtxt(sio(samples)).T
            discrete = {}
            kind_str = ['saccades', 'fixations', 'blinks']
            kind_list = [run['esacc'], run['efix'], run['eblink']]
            column_list = [info['saccade_fields'],
                           info['fixation_fields'],
                           EDF.BLINK_FIELDS]
            for s, kind, cols in zip(kind_str, kind_list, column_list):
                d = np.genfromtxt(sio(''.join(kind)), dtype=None)
                disc = dict()
                for ii, key in enumerate(cols):
                    disc[key] = d['f%s' % (ii + 1)]  # first field is junk
                discrete[s] = disc

            # parse messages
            times = list()
            msgs = list()
            for message in run['messages']:
                x = message.strip().split(None, 2)
                assert x[0] == 'MSG'
                times.append(x[1])
                msgs.append(x[2])
            discrete['messages'] = dict()
            times = np.array(times, dtype=np.float64)
            msgs = np.array(msgs, dtype='O')
            for key, val in zip(EDF.MESSAGE_FIELDS, [times, msgs]):
                discrete['messages'][key] = np.array(val)

            is_unique = len(np.unique(samples[0])) == samples.shape[1]
            if not is_unique:
                raise RuntimeError('The time stamp found has non-unique '
                                   'values. Please check your conversion '
                                   'settings and make sure not to use the '
                                   'float option.')

            # set t0 to 0 and scale to seconds
            _t_zero = samples[0, 0]
            samples[0] -= _t_zero
            samples[0] /= 1e3
            discrete['messages']['time'] -= _t_zero
            discrete['messages']['time'] /= 1e3
            info['event_types'] = []
            for kind in ['saccades', 'fixations', 'blinks']:
                df = discrete.get(kind, None)
                if df is not None:
                    for key in ('stime', 'etime'):
                        df[key] = (df[key] - _t_zero) / 1e3  # convert to fl
                    assert np.all(df['stime'] < df['etime'])
                    info['event_types'].append(kind)
            if ii < 1:
                self._t_zero = _t_zero
            info_runs.append(info)
            discrete_runs.append(discrete)
            samples_runs.append(samples)

        assert len(samples_runs) == len(discrete_runs) == len(info_runs)
        if len(samples_runs) == 1:
            self.info = info_runs[0]
            self.discrete = discrete_runs[0]
            self._samples = samples_runs[0]
        else:
            fs = info_runs[0]['sfreq']
            offsets = np.cumsum([0] + [s[0][-1] + 1. / fs
                                       for s in samples_runs[:-1]])
            for s, d, off in zip(samples_runs, discrete_runs, offsets):
                s[0] += off
                for kind in info['event_types']:
                    d[kind]['stime'] += off
                    d[kind]['etime'] += off
            discrete = dict()
            for kind in info['event_types']:
                discrete[kind] = dict()
                for col in d[kind].keys():
                    concat = np.concatenate([d[kind][col]
                                             for d in discrete_runs])
                    discrete[kind][col] = concat
            info = copy.deepcopy(info_runs[0])
            info['calibration'] = copy.deepcopy([i['calibration']
                                                 for i in info_runs])
            self.info = info
            self.discrete = discrete
            self._samples = np.concatenate(samples_runs, axis=1)
        assert self._samples.shape[0] == len(self.info['sample_fields'])
        self.info['fname'] = fname

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
