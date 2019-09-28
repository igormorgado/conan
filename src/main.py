import imageio
import numpy as np
from analysis import *
from plots import *
from synth import *


nmodes = 4
#x = build_image_sample(nmodes)
image_filename = 'imageio:camera.png'
fig, axes = plt.subplots(figsize=(6,4), dpi=150)
# image_analysis(axes, image_filename, kernel='epa')
plot_mode_trace_analysis(axes, image, dataset, optimalmodes)
plt.show()
