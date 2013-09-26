# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pylinkparse as plp
import numpy as np

fname = '../pylinkparse/tests/data/test_raw.asc'

raw = plp.Raw(fname)

raw.plot_calibration(title='9-Point Calibration')

events = np.arange(1000, 50000, 1500)

tmin, tmax = -0.2, 1.6

# epochs = plp.Epochs(raw, events=events, tmin=tmin, tmax=tmax)
