import imageio
import numpy as np
from analysis import *
from plots import *
from synth import *



image_filename = 'imageio:camera.png'
image = imageio.imread(image_filename)
distribution = image.ravel()

plt.close('all')

fig, (bef_ax, aft_ax) = plt.subplots(1,2, figsize=(8,5), dpi=150)


bef_div = make_axes_locatable(bef_ax)
bef_cb_ax = bef_div.append_axes("bottom", size='10%', pad=0.1)

aft_div = make_axes_locatable(aft_ax)
aft_cb_ax = aft_div.append_axes("bottom", size='10%', pad=0.1)

bef_ax.set_aspect(1.)
bef_ax.xaxis.set_visible(False)
bef_ax.yaxis.set_visible(False)
bef_ax.spines['right'].set_visible(False)
bef_ax.spines['bottom'].set_visible(False)
bef_ax.spines['left'].set_visible(False)
bef_ax.spines['top'].set_visible(False)
bef_ax.tick_params(direction='in')
bef_ax.tick_params(which='major', width=.5)
bef_ax.tick_params(which='minor', width=.2)

aft_ax.set_aspect(1.)
aft_ax.xaxis.set_visible(False)
aft_ax.yaxis.set_visible(False)
aft_ax.spines['right'].set_visible(False)
aft_ax.spines['bottom'].set_visible(False)
aft_ax.spines['left'].set_visible(False)
aft_ax.spines['top'].set_visible(False)
aft_ax.tick_params(direction='in')
aft_ax.tick_params(which='major', width=.5)
aft_ax.tick_params(which='minor', width=.2)

bef_im = bef_ax.imshow(image, cmap='gray')
bef_cbar = plt.colorbar(bef_im, cax=bef_cb_ax, label='Luminância', orientation='horizontal')
bef_cbar.outline.set_visible(False)

aft_im = aft_ax.imshow(image, cmap='gray', vmin=0, vmax=194)
aft_cbar = plt.colorbar(aft_im, cax=aft_cb_ax, label='Luminância', orientation='horizontal')
aft_cbar.outline.set_visible(False)

fig.savefig('camera_before_after.png', dpi=150)

plt.show()
