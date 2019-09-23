import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from KDEpy import FFTKDE
from scipy.signal import find_peaks, peak_widths

plt.close('all')

imgfile = 'imageio:camera.png'
img = imageio.imread(imgfile)
print(imgfile, np.min(img), np.max(img))


# Number of datapoints
n = 256

# data support
x = np.arange(n)

# Counting
y = np.bincount(img.ravel(), minlength=n)

# Normalize
y = y/np.sum(y)

# Find KDE
estimator = FFTKDE(kernel='gaussian', bw=4)
#estimator = FFTKDE(kernel='gaussian', bw='ISJ')
kx, ky = estimator.fit(img.ravel()).evaluate(n)

# Find KDE peaks
peaks, _ = find_peaks(ky)

# Find width at half of peak
half = peak_widths(ky, peaks, rel_height=0.5)
    
# FInd width at base of peak
full = peak_widths(ky, peaks, rel_height=1)

# Build figure and grid
fig, main_ax = plt.subplots(figsize=(5,5))


# Draw Image
im = main_ax.imshow(img, cmap='gray')
main_ax.set_aspect(1.)
main_ax.xaxis.set_visible(False)
main_ax.yaxis.set_visible(False)


divider = make_axes_locatable(main_ax)
hist_ax = divider.append_axes("bottom", 0.5, pad=0.2)
cbar_ax = divider.append_axes("right", size='5%', pad=0.1)


hist_ax.set_xlim([0,255])
hist_ax.set_ylim([0,np.max(y)])
hist_ax.set_ylabel('Density')
hist_ax.set_xlabel('Intensity')

plt.colorbar(im, cax=cbar_ax)

# Draw histogram
hist_ax.fill_between(x, y, color='#aaaaaa')

# Draw KDE
hist_ax.plot(kx, ky, '#555555')
# 
x_peaks = kx[peaks]
y_peaks = ky[peaks]
# 
# # Draw peak and widths
hist_ax.plot(x_peaks, y_peaks, 'ko', alpha=.4)

# half_width = half[0]/2
# full_width = full[0]/2
# 
# hist_ax.hlines(half[1], x_peaks - half_width, x_peaks + half_width, color='k', linestyles='dotted', alpha=.3)
# hist_ax.hlines(full[1], x_peaks - full_width, x_peaks + full_width, color='k', linestyles='dotted', alpha=.3)
# hist_ax.vlines(x_peaks, 0, 1, 'k', linestyles='dashed',  alpha=.1)



    



