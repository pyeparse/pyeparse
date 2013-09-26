# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pylinkparse as plp

fname = '../pylinkparse/tests/data/test_raw.asc'

raw = plp.Raw(fname)
print raw.info