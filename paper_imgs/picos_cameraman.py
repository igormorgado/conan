import imageio
import numpy as np
from analysis import *
from plots import *
from synth import *


def file_analysis(axes, filename, n_modes=2, vmin=None, vmax=None, bw='silverman', kernel='gaussian'):
    image = imageio.imread(image_filename)
    return image_analysis(axes, image, n_modes, vmin, vmax, bw, kernel)

def image_analysis(axes, image, n_modes=2, vmin=None, vmax=None, bw='silverman', kernel='gaussian'):
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

nmodes = 3
max_modes = 20
image_filename = 'imageio:camera.png'
image = imageio.imread(image_filename)
distribution = image.ravel()

mode_lst = analyze(distribution, max_modes=max_modes)
optimal_modes = find_optimal_modes(mode_lst, nmodes)


plt.close('all')
fig, axes = plt.subplots(figsize=(6,6), dpi=150)
image_analysis(axes, image)
#fig.savefig('cameraman_completo.png', dpi=150)

fig, axes = plt.subplots(figsize=(6,6), dpi=150)
plot_mode_trace_analysis(axes, image, mode_lst, optimal_modes)
#fig.savefig('cameraman_traco_histo.png', dpi=150)
plt.show()
