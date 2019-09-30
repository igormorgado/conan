import imageio
import numpy as np
import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import splrep, splev
from scipy.optimize import fminbound
from scipy.signal import find_peaks, peak_widths
from KDEpy import FFTKDE


def find_kde(distribution, bw='silverman', npoints=512, kernel='gaussian'):
    """ Receives a numpy array containing an image and returns
    image histogram estimatives based on Kernel density function
    with given bandwidth. The data returned are x, y datapoints"""
    estimator = FFTKDE(kernel=kernel, bw=bw)
    x, y = estimator.fit(distribution).evaluate(npoints)
    y = y[(x>=0) & (x<=255)] 
    x = x[(x>=0) & (x<=255)] 
    return (x, y), estimator.bw


def density(values, bw='silverman', npoints=512, kernel='gaussian'):
    estimator = FFTKDE(kernel=kernel, bw=bw)
    kernel_points, kernel_values = estimator.fit(values).evaluate(npoints)
    return kernel_points, kernel_values, estimator.bw



def find_kde_from_image(image, bw='silverman', npoints=512, kernel='gaussian'):
    distribution=image.ravel()
    kde_xy, bw =  find_kde(distribution=distribution, bw=bw, npoints=npoints, kernel=kernel) 
    return kde_xy, bw

def find_curves(distribution, count=2):
    """Receives a "continuous" function data and returns it's peaks
    and widths. Assumes simmetry from peaks."""
    peaks_idx, _ = find_peaks(distribution)
    peaks_idx = peaks_idx[distribution[peaks_idx].argsort()[-count:][::-1]]
    half = peak_widths(distribution, peaks_idx, rel_height=0.5)[:2]
    return (peaks_idx, half)

def find_vrange(distribution, count=2):
    pass





# PROBLEMA AQUI, as vezes retorna nenhum
def find_modes(kernel_points, kernel_values):
    return kernel_points[np.where((np.append(kernel_values[1:], -np.inf) <= kernel_values) & 
                       (kernel_values >= np.append(-np.inf, kernel_values[:-1])))]


# TODO: REVER AQUI
def find_modeid(data, bws):
    dataframe = []
    ids = np.array([1])
    mode = np.array(np.mean(data))
    for bw in bws:
        (kx, ky), _ = find_kde(data, bw=bw)
        mode_new = np.sort(find_modes(kx, ky))
        #print(f'BW: {bw:.3f} Mode_New {mode_new[:3]}')
        d = np.abs(mode_new[:, None] - mode)
        i = np.argmin(d, axis=0)
        g = np.zeros_like(mode_new)
        g[i] = ids[0:d.shape[1]]
        k = (g==0)
        g[k] = np.arange(np.sum(np.logical_not(k))+1, len(g)+1)
        ids = np.copy(g)
        mode = np.copy(mode_new)
        tmp_array = np.array([[bw]*g.shape[0], mode_new, g]).T
        dataframe.append(tmp_array)
    return dataframe


# REVER ESTA FUNCAO
def min_slope(x, y):
    order = np.argsort(x)
    # print(f'xorder: {x[order]} xshape {x.shape}')
    # print(f'yoder: {y[order]} yshape {y.shape}')
    f = splrep(x[order], y[order])
    e = (np.max(x) - np.min(x)) * 1e-4
    def df2(x, f, e):
        return ((splev(x+e, f) - splev(x-e, f))/(2*e))**2
    bw, slope, err, iters = fminbound(df2, np.min(x), np.max(x), args=(f, e), full_output=True)
    mode = splev(bw, f)
    return bw, mode, slope


def optimal_mode(dataset, modeid, max_bandwidth):
    """ Given a modeid from dataset and a maximum search bandwidth return
        the optimal bandwidth, mode value and slope
    """
    assert (modeid > 0), "ModeID should be higher than 0"
    y = dataset[(dataset[:,2] == modeid) & (dataset[:,0] <= max_bandwidth)]
    opt_bandwidth, mode, slope = min_slope(y[:,0], y[:,1])
    return opt_bandwidth, mode, slope

def find_optimal_modes(dataset, nmodes):
    X_bw, X_mode, X_id = dataset[:, 0], dataset[:, 1], dataset[:, 2]
    bw_max = np.max(X_bw[X_id == nmodes])
    optimal_modes = np.zeros((nmodes, 3), dtype='float')
    fmtstr = '#{} OptimalBW: {:.2f} Mode: {:.2f} Slope: {}'
    for i in np.arange(nmodes):
        optimal_modes[i] = optimal_mode(dataset, i+1, bw_max)
        #print(fmtstr.format(i, optimal_modes[i,0], optimal_modes[i,1], optimal_modes[i,2]))

    return optimal_modes


def plot_image(axes, image, vmin=None, vmax=None):
    im = axes.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)

    axes.set_aspect(1.)
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)

    return im

def plot_histogram(axes, distribution, nbins=256, xlabel=True, vmin=None, vmax=None):
    if vmin is None:
        vmin = 0

    if vmax is None:
        vmax = 255

    distribution = distribution[(distribution >= vmin) & (distribution <= vmax)]
    nbins = vmax - vmin + 1

    values, bins, patches = axes.hist(distribution, bins=np.arange(nbins+1), color='#a0a0a0', density=True)


    axes.set_ylim(0, None)
    #axes.set_ylim([0,np.max(values)])
    axes.set_xlim(vmin, vmax)

    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.set_facecolor('#eeeeee')

    axes.yaxis.set_label_position('left')
    axes.yaxis.set_ticks_position('right')

    axes.xaxis.set_visible(True)
    axes.xaxis.set_major_locator(MultipleLocator(50))
    axes.xaxis.set_minor_locator(MultipleLocator(25))

    axes.set_ylabel('Density')
    if xlabel:
        axes.set_xlabel('Modes')
    else:
        axes.xaxis.set_ticks([])

    axes.tick_params(direction='in', color='#ffffff')
    axes.set_axisbelow(True)

    axes.grid(which='both')
    axes.grid(which='major', color='#fefefe', linewidth=1)
    axes.grid(which='minor', color='#fefefe', linewidth=.5)

    # axes.minorticks_on()
    # axes.minorticks_on(axis='vertical')

    return values, bins

def plot_colorbar(axes, image_map):
    cbar = plt.colorbar(image_map, cax=axes, label='Luminance', orientation='horizontal')
    cbar.outline.set_visible(False)
    axes.tick_params(direction='in')
    axes.tick_params(which='major', width=.5)
    axes.tick_params(which='minor', width=.2)
    return cbar

def plot_curve(axes, x, y):
    axes.plot(x, y, '#555555')
    return axes

def plot_modelines(axes, modes, height=1, xlabel=True):
    mlines = axes.vlines(modes, 0, height, 'k', alpha=.5, linestyles='dashed')
    if xlabel:
        axes.set_xlabel('Modes')
    return axes

def plot_peaks(axes, kde_xy, peaks_idx, half, show_width=False):
    x_peaks, y_peaks = np.array(kde_xy)[:, peaks_idx]
    axes.plot(x_peaks, y_peaks, 'ko', alpha=.4)

    if show_width:
        # Assume simmetry
        half_width = half[0]/2
        #full_width = full[0]/2
        axes.hlines(half[1], 
                       x_peaks - half_width, 
                       x_peaks + half_width, 
                       color='k', linestyles='dotted', alpha=.3)
        # axes.hlines(full[1],
        #                x_peaks - full_width,
        #                x_peaks + full_width, 
        #                color='k', linestyles='dotted', alpha=.3)
    return axes

def plot_image_analysis(axes,
          image, 
          kde_xy=None, 
          peaks=None, 
          vmin=None, vmax=None,
          title=None):
    """
    Draws a full plot showing informations about the image

    Parameters:
        axes: Matplotlib axes where the plot will be drawn
        image: Image numpy array
        kde_xy: KDE points tuple (values, density)
        peaks: Peaks analysis triple (peaks indexes, half height width and full height width).
            Peaks indexes are related to kde_xy position.
        vmin: Use given vmin as limit to display the image, instead matplotlib defaults
        vmax: Use given vmax as limit to display the image, instead matplotlib defaults
    """

    # Draw Plots
    #################################################################
    divider = make_axes_locatable(axes)
    hist_ax = divider.append_axes("bottom", size='25%', pad=0.1)
    cbar_ax = divider.append_axes("bottom", size='5%', pad=0.02)

    im = plot_image(axes, image, vmin=vmin, vmax=vmax)
    distribution = image.ravel()
    plot_histogram(hist_ax, distribution, xlabel=False, vmin=vmin, vmax=vmax)

    if kde_xy is not None:
        plot_curve(hist_ax, *kde_xy)

    if kde_xy is not None and peaks is not None:
        peaks_idx, half = peaks
        plot_peaks(hist_ax, kde_xy, peaks_idx, half, show_width=True)
        # plot_modelines(hist_ax, kde_xy[0][peaks_idx], 1, xlabel=False)

    plot_colorbar(cbar_ax, im)

    if title:
        axes.set_title(title)

    return axes


def plot_mode_trace(axes, modes_dataset, minima_points):
    # Helper variables to make graph calls simpler
    mode, bw, mode_id = modes_dataset[:,1], modes_dataset[:,0], modes_dataset[:,2]
    min_mode, optimal_bw = minima_points[:,1], minima_points[:,0]

    # How to simplify this custom cmap?
    # ncolors = np.max(mode_id)
    # cmap = plt.cm.jet
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # ncmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    # bounds = np.arange(ncolors)
    # norm = mpl.colors.BoundaryNorm(bounds, ncmap.N)

    axes.set_yscale('log')
    #trc_sct = axes.scatter(mode, bw, s=1, c=mode_id, cmap=ncmap, norm=norm)
    trc_sct = axes.scatter(mode, bw, s=1)
    min_sct = axes.scatter(min_mode, optimal_bw, c='k', alpha=.6)

    axes.set_ylabel('Bandwidth')
    axes.set_xlabel('Modes')
    axes.set_ylim(np.min(bw), np.max(bw))
    axes.set_xlim(0, 255)

    return axes

def plot_mode_trace_over_histogram(axes, image, dataset, optimalmodes):
    """Shows the mode trace of a given image over the image histogram"""
    plot_histogram(axes, image.ravel())
    axes2 = axes.twinx()
    plot_mode_trace(axes2, dataset, optimalmodes)
    return axes


def pia(axes, image, nmodes=2, bw='silverman',vmin=None, vmax=None, title=None):
    kde_xy, bw = find_kde_from_image(image, bw=bw)
    peaks_idx, half =  find_curves(kde_xy[1], count=nmodes)
    plot_image_analysis(axes, image, kde_xy, (peaks_idx, half), vmin=vmin, vmax=vmax, title=title)

def pmt(axes, image, nmodes=2, bw='silverman', vmin=None, vmax=None, title=None):
    (kx, ky), bw = find_kde_from_image(image, bw=bw)
    bws = bw * np.logspace(1, -1, 201)
    modes_dataset = find_modeid(image.ravel(), bws)
    modes_dataset = np.concatenate(modes_dataset, axis=0)
    optimal_modes = find_optimal_modes(modes_dataset, nmodes=nmodes)
    plot_histogram(axes, image.ravel())
    axes2 = axes.twinx()
    plot_mode_trace(axes2, modes_dataset, optimal_modes)
    return optimal_modes


def blur(image, sigma=2):
    from scipy.ndimage.filters import gaussian_filter 
    img = np.copy(image)
    img = img.astype(np.float)
    img /= 255
    img = gaussian_filter(img, sigma=sigma)
    img *= 255
    img = img.astype(np.uint8)
    print(np.all(img == image))
    return img

plt.close('all')
# GOOD IMAGES:
image_filename = 'imageio:camera.png'
#image_filename = 'imageio:coins.png'
image_filename = 'imageio:text.png'


nmodes = 2

image = imageio.imread(image_filename)
fig, axes = plt.subplots(1,2, figsize=(12,4))
pia(axes[0], image, nmodes=nmodes)
pmt(axes[1], image, nmodes=nmodes)
fig.suptitle(f'{image_filename}')

image_blur = blur(image, 5)
fig, axes = plt.subplots(1,2, figsize=(12,4))
pia(axes[0], image_blur, nmodes=nmodes)
pmt(axes[1], image_blur,  nmodes=nmodes)
fig.suptitle(f'{image_filename} blurred')

