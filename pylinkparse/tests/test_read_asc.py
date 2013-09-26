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

    dtypes = raw._saccades.dtypes.unique()
    assert_equal(len(dtypes), 3)
    dtypes = raw._saccades.dtypes.values
    assert_equal(raw._saccades['dur'], np.int64)
    assert_equal(raw._saccades['pvl'], np.int64)
    float_fields = 'stime etime sxp syp exp eyp ampl resx resy'.split()
    dtypes = np.qunique(raw._saccades[float_fields])
    assert_equal(len(dtypes), 1)
    assert_equal(dtypes[0], np.int64)


def tets_access_data():
    raw = Raw(fname)