import imageio
import numpy as np

from analysis import *
from plots import *
from synth import *


#################################################
#
# MAIN
#
#################################################
#x = build_image_sample(nmodes)
image_filename = 'imageio:clock.png'
fig, axes = plt.subplots(figsize=(6,4), dpi=150)
image_analysis(axes, image_filename)
plt.show()

# image = imageio.imread(image_filename)
# 
# hist_xy, kde_xy, peaks = find_datapoints(image)
# dataset = analyze(image)
# optimal_modes = find_optimal_modes(dataset, n_modes)
# 
# shist(axes, image, hist_xy, kde_xy, peaks, dataset, optimal_modes)
# 
# plt.show()
#plt.close('all')
