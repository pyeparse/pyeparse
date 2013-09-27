# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from .constants import EDF
from .viz import plot_calibration, plot_heatmap_raw
import pandas as pd
import numpy as np
from datetime import datetime
from StringIO import StringIO


def _assemble_data(lines, columns, sep='[ \t]+', na_values=['.']):
    """Aux function"""
    return pd.read_table(StringIO(''.join(lines)), names=columns, sep=sep,
                         na_values=na_values)


def _extract_sys_info(line):
    return line[line.find(':'):].strip(': \n')


class Raw(object):
    """ Represent EyeLink 1000 ASCII files in Python

    Parameters
    ----------
    fname : str
        The name of the ASCII converted EDF file.
    """
    def __init__(self, fname):
        self.info = {'fname': fname, 'fields': EDF.SAMPLE.split()}

        samples, saccades, fixations, blinks, validation = \
            [[] for _ in '.....']

        preamble = []
        with open(fname, 'r') as fid:
            for line in fid:
                if not samples:
                    preamble += [line]
                if line[0].isdigit():
                    samples += [line]
                elif EDF.CODE_SAC in line:
                    saccades += [line]
                elif EDF.CODE_FIX in line:
                    fixations += [line]
                elif EDF.CODE_BLINK in line:
                    blinks += [line]

            for line in preamble:
                if '!MODE'in line:
                    line = line.split()
                    self.info['eye'] = line[-1]
                    self.info['sfreq'] = float(line[-4])
                elif 'VALIDATE' in line:
                    line = line.split()
                    xy = line[-6].split(',')
                    xy_diff = line[-2].split(',')
                    validation.append({'point-x': xy[0],
                                       'point-y': xy[1],
                                       'offset': line[-4],
                                       'diff-x': xy_diff[0],
                                       'diff-y': xy_diff[1]})
                elif 'DATE:' in line:
                    line = _extract_sys_info(line)
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
                elif 'DISPLAY_COORDS' in line:
                    self.info['screen_coords'] = np.array(line.split()[-2:],
                                                          dtype='i8')

        self.info['validation'] = pd.DataFrame(validation, dtype=np.float64)
        self._samples = _assemble_data(samples, columns=EDF.SAMPLE.split())
        [self._samples.pop(k) for k in ['N1', 'N2']]
        del samples
        self.info['event_types'] = []
        if saccades:
            self._saccades = _assemble_data(saccades, columns=EDF.SAC.split())
            del saccades
            self.info['event_types'].append('_saccades')
        if fixations:
            self._fixations = _assemble_data(fixations,
                                             columns=EDF.FIX.split())
            del fixations
            self.info['event_types'].append('_fixations')
        if blinks:
            self._blinks = _assemble_data(blinks, columns=EDF.BLINK.split())
            del blinks
            self.info['event_types'].append('_blinks')

        # set t0 to 0 and scale to seconds
        self._t_zero = self._samples['time'][0]
        for attr in ['_samples', '_saccades', '_fixations', '_blinks']:
            df = getattr(self, attr, None)
            if df:
                key = 'time' if attr == '_samples' else ['stime', 'etime']
                df[key] -= self._t_zero
                df[key] /= 1e3
                if key != 'time':
                    df['dur'] /= 1e3

    def __repr__(self):
        return '<Raw | {0} samples>'.format(len(self._samples))

    def __getitem__(self, idx):
        data = self._samples[['xpos', 'ypos', 'ps']].iloc[idx]
        times = self._samples['time']
        return data, times

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
                     title=None, show=True):
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
                         title=title, show=show)

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
