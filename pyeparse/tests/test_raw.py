import numpy as np
from numpy.testing import assert_allclose
import pytest

from pyeparse import read_raw
from pyeparse.utils import _get_test_fnames, _requires_edfapi

fnames = _get_test_fnames()


@_requires_edfapi
def test_raw_io():
    """Test raw EDF IO functionality."""
    for fi, fname in enumerate(fnames):
        raw = read_raw(fname)
        print(raw)  # test repr works

        # tests dtypes are parsed correctly that is double only
        assert raw._samples.dtype == np.float64

        if fi == 0:  # First test file has this property
            for kind in ['saccades', 'fixations', 'blinks']:
                # relax, depends on data
                assert raw.discrete[kind]['stime'][0] < 12.0
        assert raw.times[0] < 1.0
        raw.remove_blink_artifacts(use_only_blink=True)
        for interp in [None, 'zoh', 'linear']:
            raw.remove_blink_artifacts(interp)


@_requires_edfapi
def test_access_data():
    """Test raw slicing and indexing."""
    for fname in fnames:
        raw = read_raw(fname)
        for idx in [[1, 3], slice(50)]:
            data, times = raw[:, idx]
            assert data.shape[1] == len(times)
        pytest.raises(KeyError, raw.__getitem__, 'foo')
        data, times = raw[:, 1]
        assert data.ndim == 1
        assert np.atleast_1d(times).size == 1

        # test for monotonous continuity
        deltas = np.diff(raw.times)
        assert_allclose(deltas, deltas[0] * np.ones_like(deltas))
        assert_allclose(deltas[0], 0.001)
