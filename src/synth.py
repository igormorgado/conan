import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.optimize import fminbound
from scipy.signal import find_peaks, peak_widths
from KDEpy import FFTKDE


def random_modes_distribution(n, mu, sigma, p):
    return np.random.normal(mu, sigma, size=(int(n),1)).ravel()


def build_image_sample(n_modes=4, m=512, n=512):
    skew = 255//n_modes
    mu = np.linspace(skew, 255-skew, n_modes) 
    mu += np.random.randint(-skew/2, skew/2,n_modes)
    mu = mu.astype(int)
    sigma = np.random.randint(0, skew/2.2, n_modes)
    p = .2 * np.random.random_sample(n_modes) + .2
    x = random_modes_distribution(m*n, mu, sigma, p)
    return x.reshape(m, n)


def build_circle(img=None, m=512, n=512, c=127, x=None, y=None, r=None):
    if img is None:
        img = np.zeros((m,n))

    if x is None:
        x = n//2

    if y is None:
        y = m//2

    if r is None:
        r = int(((n**2 + m **2)**.5)//16)

    for i in np.arange(0, r, r//10):
        color = np.random.randint(c-10,c+10)
        cv2.circle(img, (x,y), r-i, color, -1, 8, 0) 
        print(color)

    return img

    
#oimg = np.random.randint(0,10,(512,512))
#img = np.zeros((512,512)).astype(np.uint8)
#img = build_circle(img,m=512, n=512, c=120, x=150, y=150, r=100)
#img = build_circle(img,m=512, n=512, c=100, x=350, y=350, r=100)
#print(img.dtype)
#
#distribution=img.ravel().astype(np.uint8)
#nbins = 256
#x = np.arange(nbins)
#y = np.bincount(distribution, minlength=nbins)
#y = y/np.sum(y)
#print(x.dtype)
#print(y.dtype)
#
#density = y
#fig, axes = plt.subplots()
#values, bins, patches = axes.hist(density, bins=np.arange(nbins+1), color='#a0a0a0', density=True)
#
#axes.set_ylabel('Density')
#axes.set_xlabel('Modes')
#print(np.max(density))
#axes.set_ylim(0, None)
#axes.set_xlim(0, 255)
#
#axes.spines['right'].set_visible(False)
#axes.spines['bottom'].set_visible(False)
#axes.spines['left'].set_visible(False)
#axes.spines['top'].set_visible(False)
#axes.set_facecolor('#eeeeee')
#
#axes.set_xlim([0,255])
#axes.set_ylim([0,np.max(values)])
#
#axes.yaxis.set_label_position('left')
#axes.yaxis.set_ticks_position('right')
#
#axes.xaxis.set_visible(True)
#axes.xaxis.set_major_locator(MultipleLocator(50))
#axes.xaxis.set_minor_locator(MultipleLocator(25))
#axes.xaxis.set_ticks([])
#
#axes.tick_params(direction='in', color='#ffffff')
#axes.set_axisbelow(True)
#
#axes.grid(which='both')
#axes.grid(which='major', color='#fefefe', linewidth=1)
#axes.grid(which='minor', color='#fefefe', linewidth=.5)
#
#plot_histogram(ax, y)


#img[0,0]=255
#plt.close('all')
#
#plt.imshow(img, cmap='gray' )
#plt.imshow(img, vmin=0, vmax=20, cmap='gray')
#plt.colorbar()
