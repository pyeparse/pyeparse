import numpy as np
import pandas as pd
from pylinkparse import Raw

path = 'pylinkparse/tests/data/'
fname = path + 'test_raw.asc'



def test_raw_io():
    """Test essential basic IO functionality"""
    raw = Raw(fname)
    print raw

def tets_access_data():
    raw = Raw(fname)