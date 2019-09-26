import imageio
import numpy as np
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

