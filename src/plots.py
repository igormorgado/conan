import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis import *


def plot_mode_trace(axes, mode_dataset, minima_points):
    # Helper variables to make graph calls simpler
    mode, bw, mode_id = mode_dataset[:,1], mode_dataset[:,0], mode_dataset[:,2]
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


def plot_histogram(axes, distribution):
    y, bins, patches = axes.hist(distribution, 256, color='#a0a0a0', density=True)

    axes.set_ylabel('Density')
    axes.set_xlabel('Modes')
    axes.set_ylim(0, None)
    axes.set_xlim(0, 255)

    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.set_facecolor('#eeeeee')

    axes.set_xlim([0,255])
    axes.set_ylim([0,np.max(y)])

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


# Convenience functions
def image_analysis(axes, image_filename, n_modes=3, vmin=None, vmax=None, bw='ISJ'):
    image = imageio.imread(image_filename)
    hist_xy, kde_xy, peaks = find_datapoints(image, bw=bw)
    dataset = analyze(image)
    optimalmodes = find_optimal_modes(dataset, n_modes)
    axes = shist(axes, 
                 image, 
                 hist_xy=hist_xy, 
                 kde_xy=kde_xy, 
                 peaks=peaks,
                 dataset=None,
                 optimalmodes=None,
                 vmin=None,
                 vmax=None)
    return axes



# HELPER FUNCTIONS NEED TO BE REPLACED

def all_image_analysis():
    images = [ 'imageio:camera.png',
               'imageio:checkerboard.png',
               'imageio:clock.png',
               'imageio:coins.png',
               'imageio:horse.png',
               'imageio:moon.png',
               'imageio:text.png',
               'imageio:page.png',
             ]
    for img in images:
        print(img)
        fig, axes = plt.subplots()
        fig.suptitle(img)
        axes = image_analysis(axes, img)
    

def plot_camera():
    image_filename = 'imageio:camera.png'
    fig, axes = plt.subplots(dpi=150)
    fig.suptitle(image_filename)
    axes =  image_analysis(axes, image_filename)
    return fig


def plot_mode_trace_analysis(axes, image, dataset, optimalmodes):
    plot_histogram(axes, image.ravel())
    axes2 = axes.twinx()
    plot_mode_trace(axes2, dataset, optimalmodes)
    return axes
