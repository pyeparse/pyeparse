# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pylinkparse as plp
import numpy as np

fname = '../pylinkparse/tests/data/test_raw.asc'

raw = plp.Raw(fname)

raw.plot_calibration(title='9-Point Calibration')


events = plp.find_custom_events(raw, 'user-event', event_id=1)

raw.plot_heatmap(start=10., stop=85.)

tmin, tmax, event_id = -0.5, 1.5, 1

epochs = plp.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax)

# access pandas data frame and plot single epoch
import pylab as pl
pl.figure()
epochs.data.ix[0, ['xpos', 'ypos']].plot()

# iterate over and access numpy arrays.
# find epochs withouth loss of tracking / blinks
print len([e for e in epochs if not np.isnan(e).any()])

time_mask = epochs.times > 0
times = epochs.times * 1e3

pl.figure()
pl.plot(times[time_mask], epochs[0, 0, time_mask])
pl.title('Post baseline saccade (X, pos)')
pl.show()
