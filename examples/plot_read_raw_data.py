# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pylinkparse as plp
import pylab as pl

fname = '../pylinkparse/tests/data/test_raw.asc'

raw = plp.Raw(fname)
print raw.info

raw.plot_calibration(title='9-Point Calibration')