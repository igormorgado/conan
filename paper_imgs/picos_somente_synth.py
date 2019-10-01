import imageio
import numpy as np
from analysis import *
from plots import *
from synth import *


def lhist(axes,
          image, 
          hist_xy=None,
          kde_xy=None, 
          peaks=None, 
          dataset=None,
          optimalmodes=None,
          vmin=None, vmax=None):

    return axes

nmodes = 1
maxmodes = 1

image = np.random.randint(0,10, (512, 512), np.uint8)
for x in np.arange(0, 500, 50):
    a, b = 10, 30
    image[:,x+a:x+b] = image[:,x+a:x+b]+np.arange(b-a)//2
    image[-1,-1] = 255

distribution = image.ravel()
hist_xy, kde_xy, peaks = find_datapoints(image)
mode_lst = analyze(distribution, max_modes=maxmodes)
optimalmodes = find_optimal_modes(mode_lst, nmodes)

plt.close('all')

fig, (main_ax, hist_ax) = plt.subplots(2,1,figsize=(8,6), dpi=300)

im = plot_image(main_ax, image)

divider = make_axes_locatable(hist_ax)
cbar_ax = divider.append_axes("bottom", size='10%', pad=0.02)

plot_histogram(hist_ax, distribution, xlabel=False)

plot_curve(hist_ax, *kde_xy)

peaks_idx, half, full = peaks
plot_peaks(hist_ax, kde_xy, peaks_idx, half, full, show_width=True)
plot_modelines(hist_ax, kde_xy[0][peaks_idx], 1, xlabel=False)
plt.tight_layout()
plot_colorbar(cbar_ax, im)
fig.savefig('synth_somente_pico.png', dpi=300)

fig, axes = plt.subplots(figsize=(5,4), dpi=300)
plot_mode_trace_analysis(axes, image, mode_lst, optimalmodes)
plt.tight_layout()
fig.savefig('synth_somente_traco_histo.png', dpi=300)

plt.show()
