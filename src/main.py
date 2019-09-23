import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from KDEpy import FFTKDE
from scipy.signal import find_peaks, peak_widths

plt.close('all')
#plt.style.use('classic')

def find_kde(img):
    estimator = FFTKDE(kernel='gaussian', bw='ISJ')
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

def find_datapoints(img):
    x, y = find_histogram(img)
    kx, ky = find_kde(img)
    peaks, half, full = find_curves(ky)
    x_peaks = kx[peaks]
    y_peaks = ky[peaks]
    return x, y, kx, ky, peaks, x_peaks, y_peaks,  half, full



def shist(img, datapoints, title='Sample Image'):

    # Data image extraction
    #################################################################
    x, y, kx, ky, peaks, x_peaks, y_peaks, half, full = datapoints
    half_width = half[0]/2
    full_width = full[0]/2

    # Draw Image
    #################################################################
    fig, main_ax = plt.subplots(figsize=(5,5))
    im = main_ax.imshow(img, cmap='gray')
    main_ax.set_aspect(1.)
    main_ax.xaxis.set_visible(False)
    main_ax.yaxis.set_visible(False)
    divider = make_axes_locatable(main_ax)
    hist_ax = divider.append_axes("bottom", size='15%', pad=0.2)
    cbar_ax = divider.append_axes("right", size='5%', pad=0.1)
    hist_ax.set_xlim([0,255])
    hist_ax.set_ylim([0,np.max(y)])
    hist_ax.set_ylabel('Density')
    hist_ax.set_xlabel('Intensity')
    plt.colorbar(im, cax=cbar_ax)
    main_ax.set_title(title,fontsize=16)

    # Data plots
    #################################################################

    # histogram
    hist_ax.fill_between(x, y, color='#aaaaaa')

    # KDE
    hist_ax.plot(kx, ky, '#555555')
     
    # peak and widths
    hist_ax.plot(x_peaks, y_peaks, 'ko', alpha=.4)
    # hist_ax.hlines(half[1], x_peaks - half_width, x_peaks + half_width, color='k', linestyles='dotted', alpha=.3)
    # hist_ax.hlines(full[1], x_peaks - full_width, x_peaks + full_width, color='k', linestyles='dotted', alpha=.3)
    # hist_ax.vlines(x_peaks, 0, 1, 'k', linestyles='dashed',  alpha=.1)

    return fig

def plotimg(x):
    img = imageio.imread(x)
    datapoints = find_datapoints(img)
    print(x, np.min(img), np.max(img))
    return shist(img, datapoints, title=x), datapoints

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



