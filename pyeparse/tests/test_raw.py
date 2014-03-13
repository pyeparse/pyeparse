import numpy as np
from numpy.testing import assert_allclose
import warnings
from os import path as op
from nose.tools import assert_true, assert_equal

from pyeparse import Raw

warnings.simplefilter('always')  # in case we hit warnings

path = op.join(op.split(__file__)[0], 'data')
fnames = [op.join(path, 'test_raw.asc'),
          op.join(path, 'test_2_raw.asc')]


def test_raw_io():
    """Test raw IO functionality"""
    for fi, fname in enumerate(fnames):
        raw = Raw(fname)
        print(raw)  # test repr works

        # tests dtypes are parsed correctly that is double only
        assert_equal(raw._samples.dtype, np.float64)

        if fi == 0:  # First test file has this property
            for kind in ['saccades', 'fixations', 'blinks']:
                # relax, depends on data
                assert_true(raw.discrete[kind]['stime'][0] < 5.0)
        assert_true(raw['time'][0][0] < 1.0)
        for interp in [None, 'zoh', 'linear']:
            raw.remove_blink_artifacts(interp)


def test_access_data():
    """Test raw slicing and indexing"""
    for fname in fnames:
        raw = Raw(fname)
        for idx in [[1, 3], slice(50)]:
            data, times = raw[:, idx]
            assert_equal(data.shape[1], len(times))
        data, times = raw[:, 1]
        assert_equal(data.ndim, 1)
        assert_equal(np.atleast_1d(times).size, 1)

        # test for monotonous continuity
        deltas = np.diff(raw.times)
        assert_allclose(deltas, deltas[0] * np.ones_like(deltas))
        assert_allclose(deltas[0], 0.001)
