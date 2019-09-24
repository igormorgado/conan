import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from KDEpy import FFTKDE
from scipy.signal import find_peaks, peak_widths

plt.close('all')
#plt.style.use('ggplot')

def find_kde(img, bw='ISJ'):
    estimator = FFTKDE(kernel='gaussian', bw=bw)
    x, y = estimator.fit(img.ravel()).evaluate(256)
    return x, y

def find_histogram(img):
    n = 256
    x = np.arange(n)
    y = np.bincount(img.ravel(), minlength=n)
    y = y/np.sum(y)
    return  x, y

def find_curves(data):
    peaks, _ = find_peaks(data)
    half = peak_widths(data, peaks, rel_height=0.5)
    full = peak_widths(data, peaks, rel_height=1)
    return peaks, half, full

def find_datapoints(img, bw='ISJ'):
    x, y = find_histogram(img)
    kx, ky = find_kde(img, bw=bw)
    peaks, half, full = find_curves(ky)
    x_peaks = kx[peaks]
    y_peaks = ky[peaks]
    return x, y, kx, ky, peaks, x_peaks, y_peaks,  half, full



def shist(img, datapoints, title='Sample Image', vmin=None, vmax=None):

    # Data image extraction
    #################################################################
    x, y, kx, ky, peaks, x_peaks, y_peaks, half, full = datapoints
    half_width = half[0]/2
    full_width = full[0]/2

    # Draw Plots
    #################################################################
    fig, main_ax = plt.subplots(figsize=(5,6))
    divider = make_axes_locatable(main_ax)
    hist_ax = divider.append_axes("bottom", size='25%', pad=0.1)
    cbar_ax = divider.append_axes("bottom", size='5%', pad=0.02)

    if (vmin):
        im = main_ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    else: 
        im = main_ax.imshow(img, cmap='gray')

    hist_ax.fill_between(x, y, color='#a0a0a0')
    hist_ax.plot(kx, ky, '#555555')
    hist_ax.plot(x_peaks, y_peaks, 'ko', alpha=.4)
    hist_ax.hlines(half[1], x_peaks - half_width, x_peaks + half_width, color='k', linestyles='dotted', alpha=.3)
    hist_ax.hlines(full[1], x_peaks - full_width, x_peaks + full_width, color='k', linestyles='dotted', alpha=.3)
    hist_ax.vlines(x_peaks, 0, 1, 'k', linestyles='dashed',  alpha=.1)
    cbar = plt.colorbar(im, cax=cbar_ax, label='Luminance', orientation='horizontal')

    # Customize plot
    from matplotlib.ticker import MultipleLocator

    main_ax.set_title(title,fontsize=14, fontweight='bold')
    main_ax.set_aspect(1.)
    main_ax.xaxis.set_visible(False)
    main_ax.yaxis.set_visible(False)
    main_ax.spines['right'].set_visible(False)
    main_ax.spines['bottom'].set_visible(False)
    main_ax.spines['left'].set_visible(False)
    main_ax.spines['top'].set_visible(False)

    hist_ax.set_axisbelow(True)
    hist_ax.spines['right'].set_visible(False)
    hist_ax.spines['bottom'].set_visible(False)
    hist_ax.spines['left'].set_visible(False)
    hist_ax.spines['top'].set_visible(False)
    hist_ax.set_facecolor('#eeeeee')

    hist_ax.set_xlim([0,255])
    hist_ax.set_ylim([0,np.max(y)])

    hist_ax.set_ylabel('Density')
    hist_ax.yaxis.set_label_position('left')
    hist_ax.yaxis.set_ticks_position('right')

    hist_ax.xaxis.set_visible(True)
    hist_ax.xaxis.set_major_locator(MultipleLocator(50))
    hist_ax.xaxis.set_minor_locator(MultipleLocator(25))

    hist_ax.grid(which='both')
    hist_ax.grid(which='major', color='#fefefe', linewidth=1)
    hist_ax.grid(which='minor', color='#fefefe', linewidth=.5)
    hist_ax.tick_params(direction='in', color='#ffffff')
    hist_ax.xaxis.set_ticks([])

    cbar.outline.set_visible(False)
    cbar_ax.tick_params(direction='in')
    cbar_ax.tick_params(which='major', width=.5)
    cbar_ax.tick_params(which='minor', width=.2)


    # Defining the grid and tickers in Histogram
    # hist_ax.minorticks_on()
    #hist_ax.minorticks_on(axis='vertical')
    # cbar.set_ticks([0, 50, 100, 150, 200, 250, 255])
    # cbar.set_ticklabels(['0','','','','', '250',''])

    fig.tight_layout()

    return fig


def plotimg(x, vmin=None, vmax=None, bw='ISJ'):
    img = imageio.imread(x)
    datapoints = find_datapoints(img, bw=bw)
    return shist(img, datapoints, title=x, vmin=vmin, vmax=vmax), datapoints

def plotall():
    images = [ 'imageio:camera.png',
               'imageio:checkerboard.png',
               'imageio:clock.png',
               'imageio:coins.png',
               'imageio:horse.png',
               'imageio:moon.png',
               'imageio:text.png',
               'imageio:page.png',
             ]
    for x in images:
        plotimg(x)
    
def camera():
    return plotimg('imageio:camera.png')

fig,(x, y, kx, ky, peaks, x_peaks, y_peaks, half, full) = camera()



