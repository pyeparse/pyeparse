import numpy as np
import pandas as pd
from nose.tools import assert_true
from numpy.testing import assert_equal
from pylinkparse import Raw
from pylinkparse.event import find_custom_events, Discrete


path = 'pylinkparse/tests/data/'
fname = path + 'test_raw.asc'


def test_find_custom_events():
    """Test finding user-defined events"""
    raw = Raw(fname)
    events = find_custom_events(raw, 'user-event', 1)
    assert_equal(len(events), 24)
    assert_equal(set(events[:, 1]), set([1]))
    events2 = raw.find_events('user-event', 1)
    assert_true(len(events2) > 0)
    # XXX broken since continuity is not fixed for custom events
    #     read after parsing
    # assert_array_equal(events, events2)


def test_discrete():
    """Test discrete events container"""
    dis = Discrete()
    dis.extend([np.array([1]), pd.DataFrame(), 'aaaa'])
    myrepr = '%s' % dis
    checksum = sum([int(d) for d in myrepr if d.isdigit()])
    assert_equal(checksum, 5 + len(dis))
    dis = Discrete('aaaaa')
    assert_equal(len(dis), 5)
