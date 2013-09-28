# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)


class Bunch(dict):
    """ Dict that exposes keys as attributes """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


# translate EDF definitions into a nice human-readable format
dtype_dict = dict(event='O', eye='O', time='float64', stime='float64',
                  etime='float64', xpos='float64', ypos='float64',
                  dur='float64', ps='float64', xres='float64', yres='float64',
                  sxp='float64', syp='float64', exp='float64', eyp='float64',
                  ampl='float64', pv='int64', axp='float64', ayp='float64',
                  aps='int64', status='O', msg='O')
EDF = Bunch()
EDF.CODE_PUPIL = 'PUPIL'
EDF.CODE_PUPIL_AREA = 'AREA'
EDF.CODE_PUPIL_DIAMETER = 'DIAMETER'
EDF.CODE_SSAC = 'SSACC'
EDF.CODE_ESAC = 'ESACC'
EDF.CODE_SFIX = 'SFIX'
EDF.CODE_EFIX = 'EFIX'
EDF.CODE_SBLINK = 'SBLINK'
EDF.CODE_EBLINK = 'EBLINK'
EDF.BLINK_FIELDS = ['eye', 'stime', 'etime', 'dur']  # this one is fixed
EDF.MESSAGE_FIELDS = ['time', 'msg']  # this one is fixed
EDF.START_FIELDS = ['event', 'eye', 'stime']
