import numpy as np
from numpy.testing import assert_equal
import pandas as pd
from pylinkparse import Raw

path = 'pylinkparse/tests/data/'
fname = path + 'test_raw.asc'



def test_raw_io():
    """Test essential basic IO functionality"""
    raw = Raw(fname)
    print raw  # test repr works

    # tests dtypes are parsed correctly that is double only
    dtypes = raw._samples.dtypes.unique()
    assert_equal(len(dtypes), 1)
    assert_equal(dtypes[0], np.float64)

def tets_access_data():
    raw = Raw(fname)