# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
from datetime import datetime
from cStringIO import StringIO

from .constants import EDF
from .event import find_events
from .utils import check_line_index
from .viz import plot_calibration, plot_heatmap_raw


def _assemble_data(lines, columns, sep='[ \t]+', na_values=['.']):
    """Aux function"""
    return pd.read_table(StringIO(''.join(lines)), names=columns, sep=sep,
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
            sfreq = float(line[fi + 4])
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


class Raw(object):
    """ Represent EyeLink 1000 ASCII files in Python

    Parameters
    ----------
    fname : str
        The name of the ASCII converted EDF file.
    """
    def __init__(self, fname):
        self.info = {'fname': fname}

        def_lines, samples, esacc, efix, eblink, calibs, preamble, \
            messages = [list() for _ in range(8)]
        started = False
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
                        # Add another calibration section
                        calib_lines = []
                        while True:
                            subline = fid.next()
                            if subline.startswith('START'):
                                break
                            calib_lines.append(subline)
                        calibs.append(calib_lines)
                        # let's play it safe here
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
        # parse the header
        self._parse_pramble(preamble)
        self._parse_def_lines(def_lines)

        if not all([x in self.info
                    for x in ['sample_fields', 'event_fields']]):
            raise RuntimeError('could not parse header')

        self.info['calibration'] = []
        if calibs:
            for calib_lines in calibs:
                validation = self._parse_calibration(calib_lines)
                df = pd.DataFrame(validation, dtype=np.float64)
                self.info['calibration'].append(df)
                if 'sample_fields' not in self.info:
                    self._parse_put_event_format(def_lines)
        samples = _assemble_data(samples, columns=self.info['sample_fields'])
        self.samples = samples
        d = self.discrete = {}
        kind_str = ['saccades', 'fixations', 'blinks']
        kind_list = [esacc, efix, eblink]
        column_list = [self.info['saccade_fields'],
                       self.info['fixation_fields'],
                       EDF.BLINK_FIELDS]
        for s, kind, cols in zip(kind_str, kind_list, column_list):
            d[s] = _assemble_data(check_line_index(kind), columns=cols)
        d['messages'] = _assemble_messages(messages)

        is_unique = len(self.samples['time'].unique()) == len(self.samples)
        if not is_unique:
            raise RuntimeError('The time stamp found has non-unique values. '
                               'Please check your conversion settings and '
                               'make sure not to use the float option.')

        # set t0 to 0 and scale to seconds
        self._t_zero = self.samples['time'][0]
        self.samples['time'] -= self._t_zero
        self.samples['time'] /= 1e3
        self.discrete['messages']['time'] -= self._t_zero
        self.discrete['messages']['time'] /= 1e3
        key = ['stime', 'etime', 'dur']
        self.info['event_types'] = []
        for kind in ['samples', 'saccades', 'fixations', 'blinks']:
            df = d.get(kind, None)
            if df is not None:
                if key[0] in df:  # samples do not have these keys
                    df[key[:2]] -= self._t_zero
                    df[key] /= 1e3
                self.info['event_types'].append(kind)
        df = self.samples
        self.info['data_cols'] = [kk for kk, dt in zip(df.columns, df.dtypes)
                                  if dt != 'O' and kk != 'time']

    def _parse_pramble(self, preamble_lines):
        for line in preamble_lines:
            if '!MODE'in line:
                line = line.split()
                self.info['eye'] = line[-1]
                self.info['sfreq'] = float(line[-4])
            elif 'DATE:' in line:
                line = _extract_sys_info(line).strip()
                fmt = '%a %b  %d %H:%M:%S %Y'
                self.info['meas_date'] = datetime.strptime(line, fmt)
            elif 'VERSION:' in line:
                self.info['version'] = _extract_sys_info(line)
            elif 'CAMERA:' in line:
                self.info['camera'] = _extract_sys_info(line)
            elif 'SERIAL NUMBER:' in line:
                self.info['serial'] = _extract_sys_info(line)
            elif 'CAMERA_CONFIG:' in line:
                self.info['camera_config'] = _extract_sys_info(line)

    def _parse_def_lines(self, def_lines):
        format_lines = list()
        for line in def_lines:
            if line[:7] == 'SAMPLES' or line[:6] == 'EVENTS':
                format_lines.append(line)
            elif EDF.CODE_PUPIL in line:
                if EDF.CODE_PUPIL_AREA in line:
                    self.info['ps_units'] = 'area'
                elif EDF.CODE_PUPIL_DIAMETER in line:
                    self.info['ps_units'] = 'diameter'
            elif 'PRESCALER' in line:
                k, v = line.split()
                self.info[k] = int(v)
        self._parse_put_event_format(format_lines)

    def _parse_calibration(self, calib_lines):
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
                    subline = lines.next().split()
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
                # additiona lines on our way to the
                # empty line block tail
                while True:
                    this_line = lines.next()
                    if not this_line.strip('\n'):
                        break
                    additional_lines.append(this_line)
                line += '  '
                line += '; '.join(additional_lines)
            elif 'DISPLAY_COORDS' in line:
                self.info['screen_coords'] = np.array(line.split()[-2:],
                                                      dtype='i8')

        return validations[-1]

    def _parse_put_event_format(self, def_lines):
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
                if k in self.info:
                    assert self.info[k] == v
                else:
                    self.info[k] = v
            self.info[key] = fields
        self.info['saccade_fields'] = saccade_fields
        self.info['fixation_fields'] = fixation_fields
        self.info['discretes'] = []

    def __repr__(self):
        return '<Raw | {0} samples>'.format(len(self.samples))

    def __getitem__(self, idx):
        df = self.samples
        data = df[self.info['data_cols']].values[idx]
        return data, df['time'].values

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
