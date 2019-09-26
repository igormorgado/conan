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
n_modes = 2 
#x = build_image_sample(nmodes)
image_filename = 'imageio:camera.png'
image = imageio.imread(image_filename)
dataset = analyze(image)
result = find_optimal_modes(dataset, n_modes)

#plt.close('all')
plot_test(dataset, image, result)
 # fig,(x, y, kx, ky, peaks, x_peaks, y_peaks, half, full) = camera()
# 
