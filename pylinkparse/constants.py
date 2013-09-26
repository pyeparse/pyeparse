# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)


class Bunch(dict):
    """ Dict that exposes keys as attributes """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


EDF = Bunch()

EDF.CODE_SAC = 'ESACC'
EDF.CODE_FIX = 'EFIX'
EDF.CODE_BLINK = 'EBLINK'
EDF.FIX = 'event eye stime etime dur sxp syp ps resx resy'
EDF.SAC = 'event eye stime etime dur sxp syp exp eyp ampl pvl resx resy'
EDF.BLINK = 'event eye stime etime dur'
EDF.SAMPLE = 'time xpos ypos ps xvEDF yvEDF xref yref N1 N2'
EDF.BLINK_DTYPES = 'O O int64 int64 int64'

EDF.FIX_DTYPES = ('O O float64 float64 int64 float64 float64 int64 '
                  'float64 float64')

EDF.SAC_DTYPES = ('O O float64 float64 int64 float64 float64 '
                  'float64 float64 float64 int64 float64 float64')
