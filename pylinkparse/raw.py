# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from .constants import EDF
import pandas as pd
from StringIO import StringIO


def _assemble_data(lines, columns):
    """Aux function"""
    return pd.read_data(StringIO(''.join(lines)), names=columns)


class Raw(object):
    """ Represent EyeLink 1000 ASCII files in Pyhton
    """

    def __init__(self, fname):
        self.info = {'fname': fname, 'fields': EDF.SAMPLE.split()}

        samples, saccades, fixations, blinks = [[] for _ in '....']

        with open(fname, 'r') as fid:
            for line in fid:
                if line[0].isdigit():
                    samples += [line]
                elif EDF.CODE_SAC in line:
                    saccades += [line]
                elif EDF.CODE_FIX in line:
                    fixations += [line]
                elif EDF.CODE_BLINK in line:
                    blinks += [line]
        self._saccades = _assemble_data(saccades, columns=EDF.SAC.split())
        del saccades
        self._fixations = _assemble_data(fixations, columns=EDF.FIX.split())
        del fixations
        self._blinks = _assemble_data(saccades, columns=EDF.BLINK.split())
        del blinks


    def __repr__(self):
        return '<Raw | {0}>'.format(len(self._samples))