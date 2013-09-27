import numpy as np
from nose.tools import assert_true, assert_equal
from pylinkparse import Raw
from pylinkparse.constants import EDF

path = 'pylinkparse/tests/data/'
fname = path + 'test_raw.asc'


def test_raw_io():
    """Test essential basic IO functionality"""
    raw = Raw(fname)
    print raw  # test repr works

    # tests dtypes are parsed correctly that is double only
    dtypes = raw.samples.dtypes.unique()
    assert_equal(len(dtypes), 1)
    assert_equal(dtypes[0], np.float64)

    for kind, values in raw.discrete.items():
        columns = {'saccades': EDF.SAC_DTYPES,
                   'fixations': EDF.FIX_DTYPES,
                   'blinks': EDF.BLINK_DTYPES}[kind]
        for actual, desired in zip(values.dtypes, columns.split()):
            assert_equal(actual, np.dtype(desired))

    for kind in ['saccades', 'fixations', 'blinks']:
        assert_true(raw.discrete[kind]['stime'][0] < 1.0)
    assert_true(raw.samples['time'][0] < 1.0)


def tets_access_data():
    """Test slicing and indexing"""
    raw = Raw(fname)
    for idx in [1, [1, 3], slice(50)]:
        data, times = raw[idx]
        assert_equal(len(data), len(times))

    # test for monotonous continuity
    deltas = np.unique(np.diff(times))
    assert_equal(len(deltas), 1)
    assert_equal(deltas[0], 1.0)
