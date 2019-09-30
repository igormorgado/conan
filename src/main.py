import imageio
import numpy as np
from analysis import *
from plots import *
from synth import *

def file_analysis(axes, filename, n_modes=2, vmin=None, vmax=None, bw='ISJ', kernel='gaussian'):
    image = imageio.imread(image_filename)
    return image_analysis(axes, image, n_modes, vmin, vmax, bw, kernel)

def image_analysis(axes, image, n_modes=2, vmin=None, vmax=None, bw='ISJ', kernel='gaussian'):
    hist_xy, kde_xy, peaks = find_datapoints(image, bw=bw, kernel=kernel)
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


def plot_newfig(image_filename):
    fig, axes = plt.subplots(dpi=150)
    fig.suptitle(image_filename)
    axes =  image_analysis(axes, image_filename)
    return fig


def plot_camera():
    return plot_newfig('imageio:camera.png')


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
        fig = plot_newfig(img)
    



def datalimits(data, modes=2):
    """Return the optimal minimum and maximal value based on chosen number of modes
    
    Input:
        image (np.array): 1d array
        modes (int): integer, number of expected modes

    Returns:
        tuple (int): vmin, vmax
    """
    pass


def imagelimits(image, modes=2):
    """Return the optimal minimum and maximal value based on chosen number of modes
    
    Input:
        image (np.array): Gray scale image (np.array)
        modes (int): integer, number of expected modes

    Returns:
        tuple (int): vmin, vmax
    """
    data = image.ravel()
    return datalimits(data, modes)

def filelimits(filename, modes=2):
    """Returns the optimal minimum and maximal value based on chosen number of modes
    Input:
        filename (str): Filename of fray scale image
        modes (int): integer, number of expected modes

    Returns:
        tuple (int): vmin, vmax
    """
    image = imageio.imread(filename)
    return imagelimits(image, modes)




nmodes = 4
#x = build_image_sample(nmodes)
image_filename = 'imageio:camera.png'
fig, axes = plt.subplots(figsize=(6,4), dpi=150)
image_analysis(axes, image_filename, kernel='epa')
plot_mode_trace_analysis(axes, image, dataset, optimalmodes)
plt.show()

