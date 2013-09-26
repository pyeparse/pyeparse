# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from .constants import EDF
import pandas as pd
import numpy as np
from StringIO import StringIO


def _assemble_data(lines, columns, sep=' '):
    """Aux function"""
    return pd.read_table(StringIO(''.join(lines)), names=columns, sep=sep)


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
        first_sample = 0
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
                    deg = line[-2].split(',')
                    validation.append({'point-x': xy[0], 'pint-y': xy[1],
                                       'offset': float(line[-4]),
                                       'deg-x': deg[0], 'deg-y': deg[1]})

        self.info['validation'] = pd.DataFrame(validation)
        self._samples = _assemble_data(samples, columns=EDF.SAMPLE.split())
        del samples
        if saccades:            
            self._saccades = _assemble_data(saccades, columns=EDF.SAC.split())
            del saccades
        if fixations:
            self._fixations = _assemble_data(fixations, columns=EDF.FIX.split())
            del fixations
        if blinks:
            self._blinks = _assemble_data(blinks, columns=EDF.BLINK.split())
            del blinks


    def __repr__(self):
        return '<Raw | {0} samples>'.format(len(self._samples))
