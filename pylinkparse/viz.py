# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)


def plot_calibration(raw, title='Calibration', show=True):
    """Visualize calibration

    Parameters
    ----------
    raw : instance of pylinkparse raw
        The raw object to be visualized
    title : str
        The title to be displayed.
    show : bool
        Whether to show the figure or not.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The resulting figure object
    """
    import pylab as pl
    cal = raw.info['validation']

    px, py = cal[['point-x', 'point-y']].values.T
    dx, dy = cal[['diff-x', 'diff-y']].values.T

    fig = pl.figure()
    pl.title(title)
    pl.scatter(px, py, color='gray')
    pl.scatter(px - dx, py - dy, color='red')
    if show:
        pl.show()
    return fig