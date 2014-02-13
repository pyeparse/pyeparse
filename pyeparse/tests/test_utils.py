# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import warnings
from nose.tools import assert_true

from pyeparse.utils import safe_bool

warnings.simplefilter('always')  # in case we hit warnings


def test_safe_bool():
    """ Test safe bool wrapper
    """
    df = pd.DataFrame({'a': np.arange(10)})
    f = safe_bool
    assert_true(f(df) is True)
    df.pop('a')
    assert_true(f(df) is False)

    assert_true(f(np.array([1])) is True)
    assert_true(f(np.array([])) is False)

    assert_true(f([1]) is True)
    assert_true(f([]) is False)

    assert_true(f(1) is True)
    assert_true(f(0) is False)

    assert_true(f('a') is True)
    assert_true(f('') is False)

    assert_true(f(True) is True)
    assert_true(f(False) is False)
