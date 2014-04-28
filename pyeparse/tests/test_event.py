import numpy as np
import warnings
from os import path as op
from nose.tools import assert_true, assert_raises
from numpy.testing import assert_equal, assert_array_equal

from pyeparse import Raw
from pyeparse._event import Discrete
from pyeparse.utils import _requires_edfapi

warnings.simplefilter('always')  # in case we hit warnings

fname = op.join(op.dirname(__file__), 'data', 'test_2_raw.edf')


@_requires_edfapi
def test_find_custom_events():
    """Test finding user-defined events"""
    raw = Raw(fname)
    events = raw.find_events('TRIALID', 1)
    assert_true(len(events) > 0)
    assert_raises(ValueError, raw.find_events, list(), 1)
    events_2 = raw.find_events(lambda x: 'TRIALID' in x, 1)
    assert_array_equal(events, events_2)


def test_discrete():
    """Test discrete events container"""
    dis = Discrete()
    dis.extend([np.array([1]), 'aaaa'])
    myrepr = '%s' % dis
    checksum = sum([int(d) for d in myrepr if d.isdigit()])
    assert_equal(checksum, 5 + len(dis))
    dis = Discrete('aaaaa')
    assert_equal(len(dis), 5)
