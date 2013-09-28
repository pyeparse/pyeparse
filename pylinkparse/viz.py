# Authors: Denis A. Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import math
from collections import deque
from functools import partial
from .utils import create_chunks


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


def plot_heatmap(xdata, ydata, width, height, cmap=None,
                 show=True):
    """ Plot heatmap of X/Y positions on canvas, e.g., screen

    Parameters
    ----------
    xdata : array-like
        The X position data to be visualized.
    ydata : array-like
        The Y position data to be visualized.
    width : int
        The canvas width.
    height : int
        The canvas height.
    show : bool
        Whether to show the figure or not.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The resulting figure object
    canvas : ndarray (width, height)
        The canvas including the gaze data.
    """
    import pylab as pl
    if cmap is None:
        cmap = 'RdBu_r'

    canvas = np.zeros((width, height))
    data = np.c_[xdata, ydata]
    inds = data[(data[:, 0] > 0) &
                (data[:, 1] > 0) &
                (data[:, 0] < width) &
                (data[:, 1] < height)].astype('i4')
    for x, y in inds:
        canvas[x, y] += 1
    fig = pl.figure()
    pl.imshow(canvas, extent=[0, width, 0, height],
              cmap=cmap, aspect='auto', origin='lower')

    if show:
        pl.show()
    return fig, canvas


def plot_heatmap_raw(raw, start=None, stop=None, cmap=None,
                     title=None, show=True):
    """ Plot heatmap of X/Y positions on canvas, e.g., screen

    Parameters
    ----------
    raw : instance of pylinkparse raw
        The raw object to be visualized
    start : float | None
        The canvas width.
    stop : float | None
        The canvas height.
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
    k = 'screen_coords'
    if k not in raw.info:
        raise RuntimeError('Raw object does not include '
                           'screem coordinates.')
    width, height = raw.info[k]
    if isinstance(start, float):
        start = raw.time_as_index([start])
    if isinstance(stop, float):
        stop = raw.time_as_index([stop])
    data, times = raw[start:stop]
    xdata, ydata = data[:, :2].T
    fig, _ = plot_heatmap(xdata=xdata, ydata=ydata, width=width,
                          height=height, cmap=cmap, show=False)

    if title is None:
        tstart, tstop = times[start:stop][[0, -1]]
        title = 'Raw data | {0} - {1} seconds'.format(tstart, tstop)
    pl.title(title)
    pl.xlabel('X position (px)')
    pl.ylabel('y position (px)')
    if show:
        pl.show()
    return fig

"""
Note. The following functions are based on Denis A. Engemann's and
Eric Larson's contribution to MNE-Python
"""


def figure_nobar(*args, **kwargs):
    """Make matplotlib figure with no toolbar"""
    import pylab as pl
    old_val = pl.mpl.rcParams['toolbar']
    try:
        pl.mpl.rcParams['toolbar'] = 'none'
        fig = pl.figure(*args, **kwargs)
        # remove button press catchers (for toolbar)
        for key in fig.canvas.callbacks.callbacks['key_press_event'].keys():
            fig.canvas.callbacks.disconnect(key)
    except Exception as ex:
        raise ex
    finally:
        pl.mpl.rcParams['toolbar'] = old_val
    return fig


def _prepare_trellis(n_cells, max_col):
    """Aux function
    """
    import pylab as pl
    if n_cells == 1:
        nrow = ncol = 1
    elif n_cells <= max_col:
        nrow, ncol = 1, n_cells
    else:
        nrow, ncol = int(math.ceil(n_cells / float(max_col))), max_col

    fig, axes = pl.subplots(nrow, ncol)
    axes = [axes] if ncol == nrow == 1 else axes.flatten()
    for ax in axes[n_cells:]:  # hide unused axes
        ax.set_visible(False)
    return fig, axes


def _draw_epochs_axes(epoch_idx, data, times, axes,
                      title_str, axes_handler):
    """Aux functioin"""
    this = axes_handler[0]
    for ii, data_, ax in zip(epoch_idx, data, axes):
        [l.set_data(times, d) for l, d in zip(ax.lines, data_)]
        if title_str is not None:
            ax.set_title(title_str % ii, fontsize=12)
        ax.set_ylim(data.min(), data.max())
        ax.set_yticks([])
        ax.set_xticks([])
        if vars(ax)[this]['reject'] is True:
            #  memorizing reject
            [l.set_color((0.8, 0.8, 0.8)) for l in ax.lines]
            ax.get_figure().canvas.draw()
        else:
            #  forgetting previous reject
            for k in axes_handler:
                if k == this:
                    continue
                if vars(ax).get(k, {}).get('reject', None) is True:
                    [l.set_color('k') for l in ax.lines]
                    ax.get_figure().canvas.draw()
                    break


def _epochs_navigation_onclick(event, params):
    """Aux function"""
    import pylab as pl
    p = params
    here = None
    if event.inaxes == p['back'].ax:
        here = 1
    elif event.inaxes == p['next'].ax:
        here = -1
    elif event.inaxes == p['reject-quit'].ax:
        if p['reject_idx']:
            pass
        pl.close(p['fig'])
        pl.close(event.inaxes.get_figure())

    if here is not None:
        p['idx_handler'].rotate(here)
        p['axes_handler'].rotate(here)
        this_idx = p['idx_handler'][0]
        data = p['epochs'].data[this_idx][p['picks']]
        _draw_epochs_axes(this_idx, data, p['times'], p['axes'],
                          p['title_str'],
                          p['axes_handler'])
            # XXX don't ask me why
        p['axes'][0].get_figure().canvas.draw()


def _epochs_axes_onclick(event, params):
    """Aux function"""
    reject_color = (0.8, 0.8, 0.8)
    ax = event.inaxes
    p = params
    here = vars(ax)[p['axes_handler'][0]]
    if here.get('reject', None) is False:
        idx = here['idx']
        if idx not in p['reject_idx']:
            p['reject_idx'].append(idx)
            [l.set_color(reject_color) for l in ax.lines]
            here['reject'] = True
    elif here.get('reject', None) is True:
        idx = here['idx']
        if idx in p['reject_idx']:
            p['reject_idx'].pop(p['reject_idx'].index(idx))
            good_lines = [ax.lines[k] for k in p['good_ch_idx']]
            [l.set_color('k') for l in good_lines]
            if p['bad_ch_idx'] is not None:
                bad_lines = ax.lines[-len(p['bad_ch_idx']):]
                [l.set_color('r') for l in bad_lines]
            here['reject'] = False
    ax.get_figure().canvas.draw()


def plot_epochs(epochs, epoch_idx=None, picks=None, scalings=None,
                title_str='#%003i', show=True, block=False):
    """ Visualize single trials using Trellis plot.

    Parameters
    ----------

    epochs : instance of pylinkparse.epochs.Epochs
        The epochs object
    epoch_idx : array-like | int | None
        The epochs to visualize. If None, the first 20 epochs are shown.
        Defaults to None.
    picks : array-like | None
        Channels to be included. If None only good data channels are used.
        Defaults to None
    lines : array-like | list of tuple
        Events to draw as vertical lines
    scalings : dict | None
        Scale factors for the traces. If None, defaults to:
        `dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4, emg=1e-3,
             ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)`
    title_str : None | str
        The string formatting to use for axes titles. If None, no titles
        will be shown. Defaults expand to ``#001, #002, ...``
    show : bool
        Whether to show the figure or not.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for rejecting bad trials on the fly by clicking on a
        sub plot.


    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
    """
    import pylab as pl
    if np.isscalar(epoch_idx):
        epoch_idx = [epoch_idx]
    if epoch_idx is None:
        n_events = len(epochs.events)
        epoch_idx = range(n_events)
    else:
        n_events = len(epoch_idx)
    epoch_idx = epoch_idx[:n_events]
    idx_handler = deque(create_chunks(epoch_idx, 20))

    if picks is None:
        picks = np.arange(len(epochs.info['data_cols']))
    if len(picks) < 1:
        raise RuntimeError('No appropriate channels found. Please'
                           ' check your picks')

    times = epochs.times * 1e3
    n_traces = len(picks)

    # preallocation needed for min / max scaling
    n_events = len(epochs.events)
    epoch_idx = epoch_idx[:n_events]
    idx_handler = deque(create_chunks(epoch_idx, 20))
    # handle bads

    fig, axes = _prepare_trellis(epochs._n_epochs, max_col=5)
    axes_handler = deque(range(len(idx_handler)))
    df = epochs[idx_handler[0]].data_frame
    import pdb;pdb.set_trace()
    for ii, ax in zip(idx_handler[0], axes):
        go = df.ix[ii, picks]
        go.plot(color='k', ax=ax, legend=False)
        if title_str is not None:
            ax.set_title(title_str % ii, fontsize=12)
        ax.set_ylim(go.min(), go.max())
        ax.set_yticks([])
        ax.set_xticks([])
        vars(ax)[axes_handler[0]] = {'idx': ii, 'reject': False}

    # initialize memory
    for this_view, this_inds in zip(axes_handler, idx_handler):
        for ii, ax in zip(this_inds, axes):
            vars(ax)[this_view] = {'idx': ii, 'reject': False}

    pl.tight_layout()
    navigation = figure_nobar(figsize=(3, 1.5))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 2)
    ax1 = pl.subplot(gs[0, 0])
    ax2 = pl.subplot(gs[0, 1])
    ax3 = pl.subplot(gs[1, :])

    params = {
        'fig': fig,
        'idx_handler': idx_handler,
        'epochs': epochs,
        'n_traces': n_traces,
        'picks': picks,
        'times': times,
        'scalings': scalings,
        'axes': axes,
        'back': pl.mpl.widgets.Button(ax1, 'back'),
        'next': pl.mpl.widgets.Button(ax2, 'next'),
        'reject-quit': pl.mpl.widgets.Button(ax3, 'reject-quit'),
        'title_str': title_str,
        'reject_idx': [],
        'axes_handler': axes_handler
    }
    fig.canvas.mpl_connect('button_press_event',
                           partial(_epochs_axes_onclick, params=params))
    navigation.canvas.mpl_connect('button_press_event',
                                  partial(_epochs_navigation_onclick,
                                          params=params))
    if show is True:
        pl.show(block=block)
    return fig
