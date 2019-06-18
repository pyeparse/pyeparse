import warnings
import pytest


def pytest_configure(config):
    """Configure pytest options."""
    config.addinivalue_line('usefixtures', 'matplotlib_config')


@pytest.fixture(scope='session')
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    # "force" should not really be necessary but should not hurt
    kwargs = dict()
    with warnings.catch_warnings(record=True):  # ignore warning
        matplotlib.use('agg', force=True, **kwargs)  # don't pop up windows
    import matplotlib.pyplot as plt
    assert plt.get_backend() == 'agg'
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams['figure.dpi'] = 100
