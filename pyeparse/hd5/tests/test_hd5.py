# -*- coding: utf-8 -*-
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_true
from os import path as op

from pyeparse import Raw, RawHD5
from pyeparse.utils import (_get_test_fnames, _TempDir, _requires_h5py,
                            _requires_edfapi)

temp_dir = _TempDir()

fnames = _get_test_fnames()


@_requires_edfapi
@_requires_h5py
def test_read_write_hd5():
    """Test reading and writing of HD5"""
    for fname in fnames:
        r = Raw(fname)
        out_fname = op.join(temp_dir, 'temp.hd5')
        r.save(out_fname, overwrite=True)
        assert_raises(IOError, r.save, out_fname)  # overwrite=False
        r2 = RawHD5(out_fname)
        r2.save(out_fname, overwrite=True)  # double write (make sure works)
        r2 = RawHD5(out_fname)
        # samples
        assert_array_equal(r._samples, r2._samples)
        # times
        assert_array_equal(r._times, r2._times)
        # discrete
        for key in r.discrete.keys():
            assert_array_equal(r.discrete[key], r2.discrete[key])
        # info
        assert_true(set(r.info.keys()) == set(r2.info.keys()))
        assert_array_equal(r.info['calibrations'], r2.info['calibrations'])
