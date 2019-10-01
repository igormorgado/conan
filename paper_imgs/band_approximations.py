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


def plot_curve(axes, x, y, label=None, style='solid'):
    axes.plot(x, y, '#555555', label=label, linestyle=style)
    return axes

def plot_histogram(axes, density, nbins=256, xlabel=False):
    values, bins, patches = axes.hist(density, bins=np.arange(nbins+1), color='#a0a0a0', density=True)

    axes.set_ylabel('Densidade')
    if xlabel:
        axes.set_xlabel('Modas')
    axes.set_xticklabels([])
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

def plot_colorbar(axes, image_map):
    cbar = plt.colorbar(image_map, cax=axes, label='Luminância', orientation='horizontal')
    cbar.outline.set_visible(False)
    axes.tick_params(direction='in')
    axes.tick_params(which='major', width=.5)
    axes.tick_params(which='minor', width=.2)
    return cbar



def random_modes_distribution(n, mu, sigma, p):
    return np.random.normal(mu, sigma, size=(int(n),1)).ravel()


def build_image_sample(n_modes=4, m=512, n=512):
    skew = 255//n_modes
    mu = np.linspace(skew, 255-skew, n_modes) 
    mu += np.random.randint(-skew/2, skew/2,n_modes)
    mu = mu.astype(int)
    sigma = np.random.randint(0, skew/2.2, n_modes)
    p = .2 * np.random.random_sample(n_modes) + .2
    x = random_modes_distribution(m*n, mu, sigma, p)
    return x.reshape(m, n)

def find_kde(distribution, bw='silverman', npoints=512, kernel='gaussian'):
    """ Receives a numpy array containing an image and returns
    image histogram estimatives based on Kernel density function
    with given bandwidth. The data returned are x, y datapoints"""
    estimator = FFTKDE(kernel=kernel, bw=bw)
    x, y = estimator.fit(distribution).evaluate(npoints)
    y = y[(x>=0) & (x<=255)] 
    x = x[(x>=0) & (x<=255)] 
    return (x, y), estimator.bw


plt.close('all')
image = imageio.imread('imageio:camera.png')
density = image.ravel()

fig, axes = plt.subplots(2,1,figsize=(8,4))
divider = make_axes_locatable(axes[1])
cbar_ax = divider.append_axes("bottom", size='15%', pad=0.02)

im = axes[0].imshow(image,cmap='gray')
(kx, ky), bw = find_kde(density, bw='silverman', npoints=512)
(kxx, kyy), bww = find_kde(density, bw=30, npoints=512)
axes[1].set_title('Histograma de Camera Man')
plot_histogram(axes[1], density)
plot_curve(axes[1], kx, ky,  label=r'$h={:.0f}$'.format(bw))
plot_curve(axes[1], kxx, kyy, style='dashed',  label=r'$h={:.0f}$'.format(bww))
plot_colorbar(cbar_ax, im)
axes[1].legend()


plt.show()


