import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis import *


def plot_mode_trace(axes, modes_dataset, minima_points):
    # Helper variables to make graph calls simpler
    mode, bw, mode_id = modes_dataset[:,1], modes_dataset[:,0], modes_dataset[:,2]
    min_mode, optimal_bw = minima_points[:,1], minima_points[:,0]

    # How to simplify this custom cmap?
    ncolors = np.max(mode_id)
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ncmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(ncolors)
    norm = mpl.colors.BoundaryNorm(bounds, ncmap.N)

    axes.set_yscale('log')
    trc_sct = axes.scatter(mode, bw, s=1, c=mode_id, cmap=ncmap, norm=norm)
    min_sct = axes.scatter(min_mode, optimal_bw, c='k', alpha=.6)

    axes.set_ylabel('Bandwidth')
    axes.set_xlabel('Modes')
    axes.set_ylim(np.min(bw), np.max(bw))
    axes.set_xlim(0, 255)

    return axes


def plot_histogram(axes, density, nbins=256):
    values, bins, patches = axes.hist(density, bins=np.arange(nbins+1), color='#a0a0a0', density=True)

    axes.set_ylabel('Density')
    axes.set_xlabel('Modes')
    print(np.max(density))
    axes.set_ylim(0, None)
    axes.set_xlim(0, 255)

    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.set_facecolor('#eeeeee')

    axes.set_xlim([0,255])
    axes.set_ylim([0,np.max(values)])

    axes.yaxis.set_label_position('left')
    axes.yaxis.set_ticks_position('right')

    axes.xaxis.set_visible(True)
    axes.xaxis.set_major_locator(MultipleLocator(50))
    axes.xaxis.set_minor_locator(MultipleLocator(25))
    axes.xaxis.set_ticks([])

    axes.tick_params(direction='in', color='#ffffff')
    axes.set_axisbelow(True)

    axes.grid(which='both')
    axes.grid(which='major', color='#fefefe', linewidth=1)
    axes.grid(which='minor', color='#fefefe', linewidth=.5)

    # axes.minorticks_on()
    # axes.minorticks_on(axis='vertical')

    return axes


def plot_modelines(axes, modes, height=1):
    mlines = axes.vlines(modes, 0, height, 'k', alpha=.5, linestyles='dashed')
    axes.set_xlabel('Modes')
    return axes


def plot_peaks(axes, kde_xy, peaks_idx, half, full, show_width=False):
    x_peaks, y_peaks = np.array(kde_xy)[:, peaks_idx]
    axes.plot(x_peaks, y_peaks, 'ko', alpha=.4)

    if show_width:
        # Assume simmetry
        half_width = half[0]/2
        full_width = full[0]/2
        axes.hlines(half[1], 
                       x_peaks - half_width, 
                       x_peaks + half_width, 
                       color='k', linestyles='dotted', alpha=.3)
        axes.hlines(full[1],
                       x_peaks - full_width,
                       x_peaks + full_width, 
                       color='k', linestyles='dotted', alpha=.3)
    return axes


def plot_curve(axes, x, y):
    axes.plot(x, y, '#555555')
    return axes


def plot_image(axes, image, vmin=None, vmax=None):
    if (vmin):
        im = axes.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    else: 
        im = axes.imshow(image, cmap='gray')

    axes.set_aspect(1.)
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)

    return im

def plot_colorbar(axes, image_map):
    cbar = plt.colorbar(image_map, cax=axes, label='Luminance', orientation='horizontal')
    cbar.outline.set_visible(False)
    axes.tick_params(direction='in')
    axes.tick_params(which='major', width=.5)
    axes.tick_params(which='minor', width=.2)
    return cbar


def shist(axes,
          image, 
          hist_xy=None,
          kde_xy=None, 
          peaks=None, 
          dataset=None,
          optimalmodes=None,
          vmin=None, vmax=None):
    """
    Draws a full plot showing informations about the image

    Parameters:
        axes: Matplotlib axes where the plot will be drawn
        image: Image numpy array
        hist_xy: Image histogram tuple (values, density)
        kde_xy: KDE points tuple (values, density)
        peaks: Peaks analysis triple (peaks indexes, half height width and full height width).
            Peaks indexes are related to kde_xy position.
        dataset:
        optimalmodes:
        vmin: Use given vmin as limit to display the image, instead matplotlib defaults
        vmax: Use given vmax as limit to display the image, instead matplotlib defaults
    """

    # Draw Plots
    #################################################################
    divider = make_axes_locatable(axes)
    hist_ax = divider.append_axes("bottom", size='25%', pad=0.1)
    cbar_ax = divider.append_axes("bottom", size='5%', pad=0.02)

    im = plot_image(axes, image)
    distribution = image.ravel()
    plot_histogram(hist_ax, distribution)

    if kde_xy is not None:
        plot_curve(hist_ax, *kde_xy)

    if kde_xy is not None and peaks is not None:
        peaks_idx, half, full = peaks
        plot_peaks(hist_ax, kde_xy, peaks_idx, half, full, show_width=True)
        plot_modelines(hist_ax, kde_xy[0][peaks_idx], 1)

    # TwinX here do not correcly share the hist_ax, instead
    # uses the main_ax
    # if dataset is not None:
    #     mode_ax = plt.twinx()
    #     plot_mode_trace(mode_ax, dataset, optimalmodes)

    plot_colorbar(cbar_ax, im)

    return axes


def plot_mode_trace_analysis(axes, image, dataset, optimalmodes):
    """Shows the mode trace of a given image over the image histogram"""
    plot_histogram(axes, image.ravel())
    axes2 = axes.twinx()
    plot_mode_trace(axes2, dataset, optimalmodes)
    return axes



