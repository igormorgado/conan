import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

image = imageio.imread('imageio:camera.png')

fig, axes = plt.subplots(figsize=(4,4))
divider = make_axes_locatable(axes)
im = axes.imshow(image, cmap='gray')
axes.yaxis.set_visible(False)
axes.xaxis.set_visible(False)
cbar_ax = divider.append_axes("bottom", size='5%', pad=0.05)
plt.colorbar(im, cax=cbar_ax,orientation='horizontal')
plt.tight_layout()
plt.savefig('/home/igor/cameraman_bw.png', dpi=300)


fig, axes = plt.subplots(figsize=(4,4))
divider = make_axes_locatable(axes)
im = axes.imshow(image, cmap='RdBu')
axes.yaxis.set_visible(False)
axes.xaxis.set_visible(False)
cbar_ax = divider.append_axes("bottom", size='5%', pad=0.05)
plt.colorbar(im, cax=cbar_ax,orientation='horizontal')
plt.tight_layout()
plt.savefig('/home/igor/cameraman_rdbu.png', dpi=300)


