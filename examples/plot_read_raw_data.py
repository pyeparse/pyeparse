# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pylinkparse as plp
import pylab as pl

fname = '../pylinkparse/tests/data/test_raw.asc'

raw = plp.Raw(fname)

raw.plot_calibration(title='9-Point Calibration')


events = plp.find_custom_events(raw, 'user-event')

tmin, tmax = -0.2, 1.6

# epochs = plp.Epochs(raw, events=events, tmin=tmin, tmax=tmax)
data, times = raw[:]

width = 1680
height = 1050
xdata, ydata = data.iloc[:, :2].values.T
plp.viz.plot_heatmap(xdata, ydata, width, height)
pl.title('Eye tracking heatmap')
pl.xlabel('X position (px)')
pl.ylabel('y position (px)')
