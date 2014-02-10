import numpy as np
from numpy.testing import assert_allclose
import warnings
from os import path as op
from nose.tools import assert_true, assert_equal

from pylinkparse import Raw
from pylinkparse.constants import EDF, dtype_dict

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
        dtypes = raw.samples.dtypes[:-1].unique()
        assert_equal(len(dtypes), 1)
        assert_equal(dtypes[0], np.float64)
        assert_equal(raw.samples['status'].dtype, np.object)

        saccade_dtypes = [dtype_dict[x] for x in raw.info['saccade_fields']]
        fixation_dtypes = [dtype_dict[x] for x in raw.info['fixation_fields']]
        blink_dtypes = [dtype_dict[x] for x in EDF.BLINK_FIELDS]
        message_dtypes = [dtype_dict[x] for x in EDF.MESSAGE_FIELDS]
        for kind, values in raw.discrete.items():
            columns = {'saccades': saccade_dtypes,
                       'fixations': fixation_dtypes,
                       'blinks': blink_dtypes,
                       'messages': message_dtypes}[kind]
            assert_equal(len(values.dtypes), len(columns))
            for name, actual, desired in zip(values.columns,
                                             values.dtypes, columns):
                assert_equal(actual, np.dtype(desired))

        if fi == 0:  # First test file has this property
            for kind in ['saccades', 'fixations', 'blinks']:
                # relax, depends on data
                assert_true(raw.discrete[kind]['stime'][0] < 5.0)
        assert_true(raw.samples['time'][0] < 1.0)
        for interp in [None, 'zoh', 'linear']:
            raw.remove_blink_artifacts(interp)


def test_access_data():
    """Test raw slicing and indexing"""
    for fname in fnames:
        raw = Raw(fname)
        for idx in [1, [1, 3], slice(50)]:
            data, times = raw[idx]
            assert_equal(len(data), len(times))

        # test for monotonous continuity
        deltas = np.diff(times)
        assert_allclose(deltas, deltas[0] * np.ones_like(deltas))
        assert_allclose(deltas[0], 0.001)
