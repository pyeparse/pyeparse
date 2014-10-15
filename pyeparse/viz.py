# Authors: Denis A. Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import math
from collections import deque
from functools import partial
from .utils import create_chunks, fwhm_kernel_2d
from ._fixes import string_types


def plot_raw(raw, events=None, title='Raw', show=True):
    """Visualize raw data traces

    Parameters
    ----------
    raw : instance of pyeparse raw
        The raw object to be visualized
    events : array | None
        Events associated with the Raw instance.
    title : str
        The title to be displayed.
    show : bool
        Whether to show the figure or not.

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        The resulting figure object.
    """
    import matplotlib.pyplot as mpl
    data, times = raw[:, :]
    names = raw.info['sample_fields']
    ev_x = np.array([0], int) if events is None else events[:, 0]
    fig = mpl.figure()
    n_row = len(names)
    ax0 = None
    for ii, (d, n) in enumerate(zip(data, names)):
        ax = mpl.subplot(n_row, 1, ii + 1, sharex=ax0)
        if ax0 is None:
            ax0 = ax
        ev_y = np.tile(np.array([np.min(d), np.max(d), np.nan]), len(ev_x))
        ax.plot(np.repeat(times[ev_x], 3), ev_y, color='y')
        ax.plot(times, d, color='k')
        if ii == n_row - 1:
            ax.set_xlabel('Time (sec)')
        ax.set_ylabel(n)
    if show:
        mpl.show()
    return fig


def plot_calibration(raw, title='Calibration', show=True):
    """Visualize calibration

    Parameters
    ----------
    raw : instance of pyeparse raw
        The raw object to be visualized
    title : str
        The title to be displayed.
    show : bool
        Whether to show the figure or not.

    Returns
    -------
    figs : list of of matplotlib.figure.Figure instances
        The resulting figure object
    """
    import matplotlib.pyplot as mpl
    figs = []
    for cal in raw.info['calibrations']:
        fig = mpl.figure()
        figs.append(fig)
        px, py = cal['point_x'], cal['point_y']
        dx, dy = cal['diff_x'], cal['diff_y']

        mpl.title(title)
        mpl.scatter(px, py, color='gray')
        mpl.scatter(px - dx, py - dy, color='red')
    mpl.show() if show else None
    return figs


def _plot_heatmap(xdata, ydata, width, height, cmap=None,
                  vmax=None, colorbar=True,
                  kernel=dict(size=20, half_width=10), show=True):
    """ Plot heatmap of X/Y positions on canvas"""
    import matplotlib.pyplot as mpl
    if cmap is None:
        cmap = 'RdBu_r'

    canvas = np.zeros((width, height))
    data = np.c_[xdata, ydata]
    with np.errstate(invalid='ignore'):
        mask = ((data[:, 0] > 0) & (data[:, 1] > 0) &
                (data[:, 0] < width) & (data[:, 1] < height))
    inds = data[mask].astype('i4')
    if kernel is not None:
        my_kernel = fwhm_kernel_2d(kernel['size'],
                                   kernel['half_width'])
        hsize = kernel['size'] // 2
    for x, y in inds:
        if kernel is not None:
            kern_indx = np.array([x - hsize, x + hsize])
            kern_indx[kern_indx < 0] = 0
            kern_indx = slice(*kern_indx)
            kern_indy = np.array([y - hsize, y + hsize])
            kern_indy[kern_indy < 0] = 0
            kern_indy = slice(*kern_indy)
            this_part = canvas[kern_indx, kern_indy]
            if this_part.shape == my_kernel.shape:
                this_part += my_kernel
        else:
            canvas[x, y] += 1

    fig = mpl.figure()
    if vmax is None:
        vmin = canvas.min()
        vmax = canvas.max()
    else:
        vmax = vmax
        vmin = -vmax

    # flip canvas to match width > height
    canvas = canvas.T
    mpl.imshow(canvas, extent=[0, width, 0, height],
               cmap=cmap, aspect='auto', origin='lower', vmin=vmin,
               vmax=vmax)
    mpl.colorbar() if colorbar else None
    mpl.show() if show else None
    return fig, canvas


def plot_heatmap_raw(raw, start=None, stop=None, cmap=None,
                     title=None, vmax=None,  kernel=dict(size=20, width=10),
                     show=True, colorbar=True):
    """ Plot heatmap of X/Y positions on canvas, e.g., screen

    Parameters
    ----------
    raw : instance of pyeparse raw
        The raw object to be visualized
    start : float | None
        Start time in seconds.
    stop : float | None
        End time in seconds.
    cmap : matplotlib Colormap
        The colormap to use.
    title : str
        The title to be displayed.
    vmax : float | None
        The maximum (and -minimum) value to use for the colormap.
    kernel : dict
        Parameters for the smoothing kernel (size, half_width).
    colorbar : bool
        Whether to show the colorbar.
    show : bool
        Whether to show the figure or not.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The resulting figure object
    """
    import matplotlib.pyplot as mpl
    width, height = raw.info['screen_coords']
    if isinstance(start, float):
        start = raw.time_as_index([start])[0]
    if isinstance(stop, float):
        stop = raw.time_as_index([stop])[0]
    data, times = raw[:2, start:stop]
    xdata, ydata = data
    fig, _ = _plot_heatmap(xdata=xdata, ydata=ydata, width=width,
                           height=height, cmap=cmap, vmax=vmax,
                           colorbar=False, show=False)

    if title is None:
        tstart, tstop = times[start:stop][[0, -1]]
        title = 'Raw data | {0} - {1} seconds'.format(tstart, tstop)
    mpl.title(title)
    mpl.xlabel('X position (px)')
    mpl.ylabel('y position (px)')

    mpl.colorbar() if colorbar else None
    mpl.show() if show else None
    return fig

"""
Note. The following functions are based on Denis A. Engemann's and
Eric Larson's contribution to MNE-Python
"""


def figure_nobar(*args, **kwargs):
    """Make matplotlib figure with no toolbar"""
    import matplotlib.pyplot as mpl
    old_val = mpl.rcParams['toolbar']
    try:
        mpl.rcParams['toolbar'] = 'none'
        fig = mpl.figure(*args, **kwargs)
        # remove button press catchers (for toolbar)
        keys = list(fig.canvas.callbacks.callbacks['key_press_event'].keys())
        for key in keys:
            fig.canvas.callbacks.disconnect(key)
    except Exception:
        raise
    finally:
        mpl.rcParams['toolbar'] = old_val
    return fig


def _prepare_trellis(n_cells, max_col):
    """Aux function
    """
    import matplotlib.pyplot as mpl
    if n_cells == 1:
        nrow = ncol = 1
    elif n_cells <= max_col:
        nrow, ncol = 1, n_cells
    else:
        nrow, ncol = int(math.ceil(n_cells / float(max_col))), max_col

    fig, axes = mpl.subplots(nrow, ncol)
    axes = [axes] if ncol == nrow == 1 else axes.flatten()
    for ax in axes[n_cells:]:  # hide unused axes
        ax.set_visible(False)
    return fig, axes


def _draw_epochs_axes(epoch_idx, data, times, axes,
                      title_str, axes_handler, discretes,
                      discrete_colors):
    """Aux functioin"""
    this = axes_handler[0]
    data = np.ma.masked_invalid(data)
    import matplotlib.pyplot as mpl
    for ii, data_, ax in zip(epoch_idx, data, axes):
        [l.set_data(times, d) for l, d in zip(ax.lines, data_)]
        n_disc_lines = 0
        if discrete_colors is not None:
            color = discrete_colors[ii]
        else:
            color = 'orange'
        if discretes is not None:
            if discretes[ii] is not None:
                for here in discretes[ii]:
                    ax.axvline(here, color=color, linestyle='--')
                    n_disc_lines += 1
                    vars(ax.lines[-1])['def-col'] = color
        if title_str is not None:
            _set_title(ax, title_str, ii)
        ax.set_ylim(data.min(), data.max())
        if ii % 5:
            [l.set_visible(0) for l in ax.get_yticklabels()]
        if ii < len(epoch_idx) - 5:
            [l.set_visible(0) for l in ax.get_xticklabels()]
        else:
            [l.set_fontsize(8) for l in ax.get_xticklabels()]
            [l.set_fontsize(8) for l in ax.get_yticklabels()]
            labels = ax.get_xticklabels()
            mpl.setp(labels, rotation=45)
        ax.get_figure().canvas.draw()
        vars(ax)[this]['n_disc_lines'] = n_disc_lines
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
                    [l.set_color(vars(l)['def-col']) for l in ax.lines]
                    ax.get_figure().canvas.draw()
                    break


def _epochs_navigation_onclick(event, params):
    """Aux function"""
    import matplotlib.pyplot as mpl
    p = params
    here = None
    if event.inaxes == p['back'].ax:
        here = 1
    elif event.inaxes == p['next'].ax:
        here = -1
    elif event.inaxes == p['reject-quit'].ax:
        if p['reject_idx']:
            pass
        mpl.close(p['fig'])
        mpl.close(event.inaxes.get_figure())

    if here is not None and len(p['axes_handler']) > 1:
        before = p['axes_handler'][0]
        for ax in p['axes']:
            assert all([ii in vars(ax) for ii in p['axes_handler']])
            dd = -vars(ax)[before]['n_disc_lines']
            if dd:
                del ax.lines[dd:]
                ax.get_figure().canvas.draw()
            dd = 0
        p['idx_handler'].rotate(here)
        p['axes_handler'].rotate(here)
        this_idx = p['idx_handler'][0]
        data = p['epochs'].data[this_idx][:, p['picks']]
        _draw_epochs_axes(this_idx, data, p['times'], p['axes'],
                          p['title_str'],
                          p['axes_handler'],
                          p['discretes'],
                          p['discrete_colors'])
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
            for line in ax.lines:
                vars(line)['def-col'] = line.get_color()
                line.set_color(reject_color)
            here['reject'] = True
    elif here.get('reject', None) is True:
        idx = here['idx']
        if idx in p['reject_idx']:
            p['reject_idx'].pop(p['reject_idx'].index(idx))
            for line in ax.lines:
                line.set_color(vars(line)['def-col'])
            here['reject'] = False
    ax.get_figure().canvas.draw()


def _set_title(ax, title_str, ii):
    """Handle titles"""
    if isinstance(title_str, string_types):
        title = title_str % ii
    elif title_str is None:
        title = '#%00i' % ii
    else:
        title = title_str[ii]
    ax.set_title(title, fontsize=12)


def plot_epochs(epochs, epoch_idx=None, picks=None, n_chunks=20,
                title_str='#%003i', show=True, draw_discrete=None,
                discrete_colors=None, block=False):
    """ Visualize single trials using Trellis plot.

    Parameters
    ----------

    epochs : instance of pyeparse.Epochs
        The epochs object
    epoch_idx : array-like | int | None
        The epochs to visualize. If None, the first 20 epochs are shown.
        Defaults to None.
    n_chunks : int
        The number of chunks to use for display.
    picks : array-like | None
        Channels to be included. If None only good data channels are used.
        Defaults to None
    lines : array-like | list of tuple
        Events to draw as vertical lines
    title_str : None | str | list-like
        The string formatting to use for axes titles. If None, no titles
        will be shown. Defaults expand to ``#001, #002, ...``. If list-like,
        must be of same length as epochs.events.
    show : bool
        Whether to show the figure or not.
    draw_discrete : {saccades, blinks, fixations} | list-like | None |
        The events to draw as vertical lines.
    discrete_colors: : list-like | None
        list of str or color objects with length of discrete events drawn.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for rejecting bad trials on the fly by clicking on a
        sub plot.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as mpl
    if np.isscalar(epoch_idx):
        epoch_idx = [epoch_idx]
    if epoch_idx is None:
        n_events = len(epochs.events)
        epoch_idx = range(n_events)
    else:
        n_events = len(epoch_idx)

    if picks is None:
        picks = np.arange(len(epochs.info['data_cols']))
    elif all(p in epochs.ch_names for p in picks):
        # epochs.data does not include time
        ch_names = [ch for ch in epochs.ch_names if ch in
                    epochs.info['data_cols']]
        picks = [ch_names.index(k) for k in picks]
    elif any(p not in epochs.ch_names and isinstance(p, string_types)
             for p in picks):
        wrong = [p for p in picks if p not in epochs.ch_names]
        raise ValueError('Some channels are not defined: ' + '; '.join(wrong))
    if len(picks) < 1:
        raise RuntimeError('No appropriate channels found. Please'
                           ' check your picks')
    if discrete_colors is not None:
        if len(discrete_colors) != len(epochs.events):
            raise ValueError('The length of `discrete_colors` must equal '
                             'the number of epochs.')

    times = epochs.times * 1e3
    n_traces = len(picks)

    # preallocation needed for min / max scaling
    n_events = len(epochs.events)
    epoch_idx = epoch_idx[:n_events]
    idx_handler = deque(create_chunks(epoch_idx, n_chunks))
    # handle bads
    this_idx = idx_handler[0]
    fig, axes = _prepare_trellis(len(this_idx), max_col=5)
    axes_handler = deque(range(len(idx_handler)))
    data = np.ma.masked_invalid(epochs.data[this_idx][:, picks])
    if isinstance(draw_discrete, string_types):
        key = dict()
        for k in epochs.info['discretes']:
            key[k.strip('_')] = k
        key = key[draw_discrete]
        discretes = [d['stime'] * 1e3 for d in vars(epochs)[key]
                     if d is not None]
    elif draw_discrete is None:
        discretes = None
    else:
        discretes = draw_discrete

    labelfontsize = 10
    for i_ax, (ii, ax, data_) in enumerate(zip(this_idx, axes, data)):
        ax.plot(times, data_.T, color='k')
        ax.axvline(0.0, color='gray')
        vars(ax.lines[-1])['def-col'] = 'gray'
        n_disc_lines = 0
        if discrete_colors is not None:
            color = discrete_colors[ii]
        else:
            color = 'orange'
        if discretes is not None:
            if discretes[ii] is not None:
                for here in discretes[ii]:
                    ax.axvline(here, color=color, linestyle='--')
                    n_disc_lines += 1
                    vars(ax.lines[-1])['def-col'] = color
        if title_str is not None:
            _set_title(ax, title_str, ii)
        ax.set_ylim(data.min(), data.max())
        if i_ax % 5:
            [l.set_visible(0) for l in ax.get_yticklabels()]
        else:
            [l.set_fontsize(labelfontsize) for l in ax.get_yticklabels()]

        if i_ax < len(this_idx) - 5:
            [l.set_visible(0) for l in ax.get_xticklabels()]
        else:
            [l.set_fontsize(labelfontsize) for l in ax.get_xticklabels()]
            labels = ax.get_xticklabels()
            mpl.setp(labels, rotation=45)
        vars(ax)[axes_handler[0]] = {'idx': ii, 'reject': False,
                                     'n_disc_lines': n_disc_lines}

    # XXX find smarter way to to tight layout for incomplete plots
    # fig.tight_layout()

    # initialize memory
    for this_view, this_inds in zip(axes_handler, idx_handler):
        if this_view > 0:
            # all other views than the current one
            for ii, ax in enumerate(axes):
                vars(ax)[this_view] = {'idx': ii, 'reject': False,
                                       'n_disc_lines': 0}

    navigation = figure_nobar(figsize=(3, 1.5))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 2)
    ax1 = mpl.subplot(gs[0, 0])
    ax2 = mpl.subplot(gs[0, 1])
    ax3 = mpl.subplot(gs[1, :])

    params = {
        'fig': fig,
        'idx_handler': idx_handler,
        'epochs': epochs,
        'n_traces': n_traces,
        'picks': picks,
        'times': times,
        'axes': axes,
        'back': mpl.Button(ax1, 'back'),
        'next': mpl.Button(ax2, 'next'),
        'reject-quit': mpl.Button(ax3, 'reject-quit'),
        'title_str': title_str,
        'reject_idx': [],
        'axes_handler': axes_handler,
        'discretes': discretes,
        'discrete_colors': discrete_colors,
    }
    fig.canvas.mpl_connect('button_press_event',
                           partial(_epochs_axes_onclick, params=params))
    navigation.canvas.mpl_connect('button_press_event',
                                  partial(_epochs_navigation_onclick,
                                          params=params))
    mpl.show(block=block) if show else None
    return fig
