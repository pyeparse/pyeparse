# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
from datetime import datetime
try:
    from cStringIO import StringIO as sio
except ImportError:  # py3 has renamed this
    from io import StringIO as sio

from .constants import EDF
from .event import find_events
from .utils import check_line_index, safe_bool, next
from .viz import plot_calibration, plot_heatmap_raw


def _assemble_data(lines, columns, sep='[ \t]+', na_values=['.']):
    """Aux function"""
    return pd.read_table(sio(''.join(lines)), names=columns, sep=sep,
                         na_values=na_values)


def _assemble_messages(lines):
    """Aux function for dealing with messages (sheesh)"""
    messages = list()
    for line in lines:
        line = line.strip().split(None, 2)
        assert line[0] == 'MSG'
        messages.append(line[1:3])
    return pd.DataFrame(messages, columns=EDF.MESSAGE_FIELDS,
                        dtype=(np.float64, 'O'))


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


def _merge_run_data(run1, run2):
    """Merge two runs -- use with reduce"""
    if run1[2]['sfreq'] != run2[2]['sfreq']:
        raise RuntimeError('Sample frequencies differ across runs')
    if safe_bool(run1[0].columns != run2[0].columns):
        raise RuntimeError('Sample columns differ across runs')
    sfreq = run1[2]['sfreq']
    offset = run1[0]['time'].values[-1] + (1. / sfreq)
    run2[0]['time'] += offset
    samples = pd.concat([run1[0], run2[0]], ignore_index=True)
    diff = np.diff(samples.time.values)
    diff = np.ma.masked_invalid(diff).astype(np.int64)
    diff = np.unique(diff)
    if len(diff) > 1:
        raise RuntimeError('Could not concatenate runs')
    discrete = {}
    for ((kind1, data1), (kind2, data2)) in zip(run1[1].items(),
                                                run2[1].items()):
        if 'stime' in data2:
            data2['stime'] += offset
            data2['etime'] += offset
        elif 'time' in data2:
            data2['time'] += offset
        discrete.update({kind1: pd.concat([data1, data2],
                                          ignore_index=True)})
    info = run1[2]  # let's keep it simple for the moment and assume
                    # infos don't differ
    info['calibration'] = [i['calibration'] for i in [run1[2], run2[2]]]
    return [samples, discrete, info]


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
                        # deal with old pandas version, add an index.
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
                df = pd.DataFrame(validation, dtype=np.float64)
                info['calibration'] = df
                if 'sample_fields' not in info:
                    _parse_put_event_format(info, run['def_lines'])
            samples = _assemble_data(run['samples'],
                                     columns=info['sample_fields'])
            del run['samples']
            discrete = {}
            kind_str = ['saccades', 'fixations', 'blinks']
            kind_list = [run['esacc'], run['efix'], run['eblink']]
            column_list = [info['saccade_fields'],
                           info['fixation_fields'],
                           EDF.BLINK_FIELDS]
            for s, kind, cols in zip(kind_str, kind_list, column_list):
                discrete[s] = _assemble_data(check_line_index(kind),
                                             columns=cols)
            discrete['messages'] = _assemble_messages(run['messages'])

            is_unique = len(samples['time'].unique()) == len(samples)
            if not is_unique:
                raise RuntimeError('The time stamp found has non-unique '
                                   'values. Please check your conversion '
                                   'settings and make sure not to use the '
                                   'float option.')

            # set t0 to 0 and scale to seconds
            _t_zero = samples['time'][0]
            samples['time'] -= _t_zero
            samples['time'] /= 1e3
            discrete['messages']['time'] -= _t_zero
            discrete['messages']['time'] /= 1e3
            info['event_types'] = []
            for kind in ['saccades', 'fixations', 'blinks']:
                df = discrete.get(kind, None)
                if df is not None:
                    df[['stime', 'etime']] -= _t_zero
                    df[['stime', 'etime', 'dur']] /= 1e3
                    info['event_types'].append(kind)
            if ii < 1:
                self._t_zero = _t_zero
            df = samples
            info['data_cols'] = [kk for kk, dt in zip(df.columns, df.dtypes)
                                 if dt != 'O' and kk != 'time']
            info_runs.append(info)
            discrete_runs.append(discrete)
            samples_runs.append(samples)

        assert len(samples_runs) == len(discrete_runs) == len(info_runs)
        if len(samples_runs) == 1:
            self.info = info_runs[0]
            self.samples = samples_runs[0]
            self.discrete = discrete_runs[0]
        else:
            one_run = None
            # instead of using "reduce"
            for run in zip(samples_runs, discrete_runs, info_runs):
                if one_run is None:
                    one_run = run
                else:
                    one_run = _merge_run_data(one_run, run)
            self.samples, self.discrete, self.info = one_run
        self.info['fname'] = fname

    def __repr__(self):
        return '<Raw | {0} samples>'.format(len(self.samples))

    def __getitem__(self, idx):
        df = self.samples
        data = df[self.info['data_cols']].values[idx]
        return data, df['time'].values[idx]

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
            The canvas width.
        stop : float | None
            The canvas height.
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
        raw : instance of pylinkparse.raw.Raw
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
