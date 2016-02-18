# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import pyeparse as pp

fname = '../pyeparse/tests/data/test_raw.edf'

raw = pp.read_raw(fname)

# visualize initial calibration
raw.plot_calibration(title='5-Point Calibration')

# create heatmap
raw.plot_heatmap(start=3., stop=60.)

# find events and epoch data
events = raw.find_events('SYNCTIME', event_id=1)
tmin, tmax, event_id = -0.5, 1.5, 1
epochs = pp.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                   tmax=tmax)

# access pandas data frame and plot single epoch
fig, ax = plt.subplots()
ax.plot(epochs[3].get_data('xpos')[0], epochs[3].get_data('ypos')[0])

# iterate over and access numpy arrays.
# find epochs withouth loss of tracking / blinks
print(len([e for e in epochs if not np.isnan(e).any()]))

fig, ax = plt.subplots()
ax.set_title('Superimposed saccade responses')
n_trials = 12  # first 12 trials
for epoch in epochs[:n_trials]:
    ax.plot(epochs.times * 1e3, epoch[0].T)

time_mask = epochs.times > 0
times = epochs.times * 1e3

fig, ax = plt.subplots()
ax.plot(times[time_mask], epochs.data[0, 0, time_mask])
ax.set_title('Post baseline saccade (X, pos)')

# plot single trials
epochs.plot(picks=['xpos'], draw_discrete='saccades')
plt.show()
