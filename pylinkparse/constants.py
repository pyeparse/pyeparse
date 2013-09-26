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

EDF.FIX = 'eye stime etime dur axp ayp aps'
EDF.SAC = 'eye stime etime dur sxp syp exp eyp ampl'
EDF.BLINK = 'eye stime etime dur'
EDF.SAMPLE = 'time xpos ypos ps xvEDF yvEDF xref yref N1 N2'
