import numpy as np
from scipy.interpolate import splrep, splev
from scipy.optimize import fminbound
from scipy.signal import find_peaks, peak_widths
from KDEpy import FFTKDE


def find_kde(distribution, bw='silverman', npoints=512, kernel='gaussian'):
    """ Receives a numpy array containing an image and returns
    image histogram estimatives based on Kernel density function
    with given bandwidth. The data returned are x, y datapoints"""
    estimator = FFTKDE(kernel=kernel, bw=bw)
    x, y = estimator.fit(distribution).evaluate(npoints)

    # Fix silverman bias in small datasets
    if (bw == 'silverman') and (estimator.bw < 1):
        estimator = FFTKDE(kernel=kernel, bw=1)
        x, y = estimator.fit(distribution).evaluate(npoints)

    y = y[(x>=0) & (x<=255)] 
    x = x[(x>=0) & (x<=255)] 
    return (x, y)

def find_kde_from_image(image, bw='silverman', npoints=512, kernel='gaussian'):
    distribution=image.ravel()
    return find_kde(distribution=distribution, bw=bw, npoints=npoints, kernel=kernel) 

def find_kde_from_file(filename, bw='silverman', npoints=512, kernel='gaussian'):
    image = imageio.imread(filename)
    return find_kde_from_image(image=image, bw=bw, npoints=npoints, kernel=kernel)


def find_histogram(distribution, nbins=256):
    """ Receives a numpy array containing an image and returns
    an histogram data x,y datapoints. """
    x = np.arange(nbins)
    y = np.bincount(distribution, minlength=nbins)
    y = y/np.sum(y)
    return  (x, y)

def find_histogram_from_image(image, nbins=256):
    distribution=image.ravel().astype(np.uint8)
    return find_histogram(distribution=distribution, nbins=nbins)


def find_histogram_from_file(filename, nbins=256):
    image = imageio.imread(filename)
    return find_histogram_from_image(image=image, nbins=nbins)


def find_curves(distribution):
    """Receives a "continuous" function data and returns it's peaks
    and widths. Assumes simmetry from peaks."""
    peaks_idx, _ = find_peaks(distribution, distance=5 )
    half = peak_widths(distribution, peaks_idx, rel_height=0.5)[:2]
    full = peak_widths(distribution, peaks_idx, rel_height=1)[:2]
    return (peaks_idx, half, full)


def find_datapoints(img, bw='silverman', kernel='gaussian'):
    hst_xy = find_histogram(img.ravel())
    kde_xy = find_kde(img.ravel(), bw=bw, kernel=kernel)
    peaks = find_curves(kde_xy[1])
    return hst_xy, kde_xy, peaks


def density(values, bw='silverman', npoints=512, kernel='gaussian'):
    estimator = FFTKDE(kernel=kernel, bw=bw)
    x, y = estimator.fit(values).evaluate(npoints)
    y = y[(x>=0) & (x<=255)] 
    x = x[(x>=0) & (x<=255)] 
    return x, y, estimator.bw


# PROBLEMA AQUI, as vezes retorna nenhum
def find_modes(kernel_points, kernel_values):
    return kernel_points[np.where((np.append(kernel_values[1:], -np.inf) <= kernel_values) & 
                       (kernel_values >= np.append(-np.inf, kernel_values[:-1])))]


# TODO: REVER AQUI
def find_modeid(data, bws):
    dataframe = []
    ids = np.array([1])
    mode = np.array(np.mean(data))
    for bw in bws:
        kx, ky, _ = density(data, bw=bw)
        mode_new = np.sort(find_modes(kx, ky))
        #print(f'BW: {bw:.3f} Mode_New {mode_new[:3]}')
        d = np.abs(mode_new[:, None] - mode)
        i = np.argmin(d, axis=0)
        g = np.zeros_like(mode_new)
        g[i] = ids[0:d.shape[1]]
        k = (g==0)
        g[k] = np.arange(np.sum(np.logical_not(k))+1, len(g)+1)
        ids = np.copy(g)
        mode = np.copy(mode_new)
        tmp_array = np.array([[bw]*g.shape[0], mode_new, g]).T
        dataframe.append(tmp_array)

     
    return dataframe

# REVER ESTA FUNCAO
def min_slope(x, y):
    order = np.argsort(x)
    # print(f'xorder: {x[order]} xshape {x.shape}')
    # print(f'yoder: {y[order]} yshape {y.shape}')
    f = splrep(x[order], y[order])
    e = (np.max(x) - np.min(x)) * 1e-4
    def df2(x, f, e):
        return ((splev(x+e, f) - splev(x-e, f))/(2*e))**2
    bw, slope, err, iters = fminbound(df2, np.min(x), np.max(x), args=(f, e), full_output=True)
    mode = splev(bw, f)
    return bw, mode, slope


def optimal_mode(dataset, modeid, max_bandwidth):
    """ Given a modeid from dataset and a maximum search bandwidth return
        the optimal bandwidth, mode value and slope
    """
    assert (modeid > 0), "ModeID should be higher than 0"
    y = dataset[(dataset[:,2] == modeid) & (dataset[:,0] <= max_bandwidth)]
    opt_bandwidth, mode, slope = min_slope(y[:,0], y[:,1])
    return opt_bandwidth, mode, slope


def analyze(distribution, max_modes=20):
    # Initial bw estimator based on Silverman
    dist = distribution.ravel()

    kx, ky, bw = density(dist)
    bws = bw * np.logspace(1, -1, 201)

    # Extraction of modes and bandwidths
    mode_lst = find_modeid(dist, bws)
    mode_lst = np.concatenate(mode_lst, axis=0)
    mode_lst = mode_lst[mode_lst[:,2] <= max_modes]

    return mode_lst

def find_optimal_modes(dataset, n_modes):
    X_bw, X_mode, X_id = dataset[:, 0], dataset[:, 1], dataset[:, 2]
    bw_max = np.max(X_bw[X_id == n_modes])
    optimal_modes = np.zeros((n_modes, 3), dtype='float')
    fmtstr = 'Mode ID: {} OptimalBW: {:.2f} Mode: {:.2f} Slope: {}'
    for i in np.arange(n_modes):
        optimal_modes[i] = optimal_mode(dataset, i+1, bw_max)
        # print(fmtstr.format(i+1,*(optimal_modes[i])))

    return optimal_modes

