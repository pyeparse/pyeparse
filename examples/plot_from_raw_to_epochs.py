# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pylinkparse as plp
import numpy as np

fname = '../pylinkparse/tests/data/test_raw.asc'

raw = plp.Raw(fname)

# visualize initial calibration
raw.plot_calibration(title='9-Point Calibration')

# create heatmap
raw.plot_heatmap(start=10., stop=85.)

# find events and epoch data
events = raw.find_events('user-event', event_id=1)
tmin, tmax, event_id = -0.5, 1.5, 1
epochs = plp.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax)

# access pandas data frame and plot single epoch
import pylab as pl
pl.figure()
epochs.data_frame.ix[3, ['xpos', 'ypos']].plot()

# iterate over and access numpy arrays.
# find epochs withouth loss of tracking / blinks
print len([e for e in epochs if not np.isnan(e).any()])

pl.figure()
pl.title('Superimposed saccade responses')
n_trials = 12  # first 12 trials
for epoch in epochs[:n_trials]:
    pl.plot(epochs.times * 1e3, epoch[0].T)
pl.show()

time_mask = epochs.times > 0
times = epochs.times * 1e3

pl.figure()
pl.plot(times[time_mask], epochs.data[0, 0, time_mask])
pl.title('Post baseline saccade (X, pos)')
pl.show()

# plot single trials
epochs.plot(picks=['xpos'], draw_events='saccades')
