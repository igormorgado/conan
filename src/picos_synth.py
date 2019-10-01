import imageio
import numpy as np
from analysis import *
from plots import *
from synth import *


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

fig, axes = plt.subplots(figsize=(6,6), dpi=150)
shist(axes, image, hist_xy=hist_xy, kde_xy=kde_xy, peaks=peaks)
#fig.savefig('synth_completo.png', dpi=150)

fig, axes = plt.subplots(figsize=(6,6), dpi=150)
plot_mode_trace_analysis(axes, image, mode_lst, optimalmodes)
#fig.savefig('synth_traco_histo.png', dpi=150)

#plt.show()
