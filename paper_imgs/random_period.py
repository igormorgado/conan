import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


plt.close('all')

image = np.random.randint(0,10, (512, 512), np.uint8)
for x in np.arange(0, 500, 50):
    a, b = 10, 30
    image[:,x+a:x+b] = image[:,x+a:x+b]+np.arange(b-a)//2
    image[-1,-1] = 255

fig, axes = plt.subplots(figsize=(4,4))
divider = make_axes_locatable(axes)
cbar_ax = divider.append_axes("bottom", size='5%', pad=0.05)
axes.yaxis.set_visible(False)
axes.xaxis.set_visible(False)

im = axes.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.colorbar(im, cax=cbar_ax,orientation='horizontal')

plt.tight_layout()
plt.savefig('/home/igor/high_band_reg.png', dpi=300)


fig, axes = plt.subplots(figsize=(4,4))
divider = make_axes_locatable(axes)
cbar_ax = divider.append_axes("bottom", size='5%', pad=0.05)
axes.yaxis.set_visible(False)
axes.xaxis.set_visible(False)
im = axes.imshow(image, cmap='gray',vmin=0, vmax=20)
plt.colorbar(im, cax=cbar_ax,orientation='horizontal')
plt.tight_layout()
plt.savefig('/home/igor/high_band_lim.png', dpi=300)


fig, axes = plt.subplots(2,1, figsize=(8,4))
#divider = make_axes_locatable(axes[1])
hist_ax = axes[1]
divider = make_axes_locatable(hist_ax)


im = axes[0].imshow(image, cmap='gray')
distribution = image.ravel()
nbins = 255

values, bins, patches = hist_ax.hist(distribution, nbins, color='#a0a0a0', density=True)

hist_ax.set_ylim(0, None)
hist_ax.set_xlim(0, 255)
hist_ax.spines['right'].set_visible(False)
hist_ax.spines['bottom'].set_visible(False)
hist_ax.spines['left'].set_visible(False)
hist_ax.spines['top'].set_visible(False)
hist_ax.set_facecolor('#eeeeee')
hist_ax.yaxis.set_label_position('left')
hist_ax.yaxis.set_ticks_position('right')
hist_ax.xaxis.set_visible(True)
#hist_ax.xaxis.set_major_locator(MultipleLocator(50))
#hist_ax.xaxis.set_minor_locator(MultipleLocator(25))
#hist_ax.set_xticklabels([])
hist_ax.set_ylabel('Density')
hist_ax.tick_params(direction='in', color='#ffffff')
hist_ax.set_axisbelow(True)
hist_ax.grid(which='both')
hist_ax.grid(which='major', color='#fefefe', linewidth=1)
hist_ax.grid(which='minor', color='#fefefe', linewidth=.5)


plt.savefig('/home/igor/high_band_band.png', dpi=300)
