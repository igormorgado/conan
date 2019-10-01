import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import splrep, splev
from scipy.optimize import fminbound
from scipy.signal import find_peaks, peak_widths
from KDEpy import FFTKDE

def find_modes(kernel_points, kernel_values):
    return kernel_points[np.where((np.append(kernel_values[1:], -np.inf) <= kernel_values) & 
                        (kernel_values >= np.append(-np.inf, kernel_values[:-1])))]


def density(values, bw='silverman', npoints=512, kernel='gaussian'):
    estimator = FFTKDE(kernel=kernel, bw=bw)
    kernel_points, kernel_values = estimator.fit(values).evaluate(npoints)
    return kernel_points, kernel_values, estimator.bw

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

def find_optimal_modes(dataset, nmodes):
    X_bw, X_mode, X_id = dataset[:, 0], dataset[:, 1], dataset[:, 2]
    bw_max = np.max(X_bw[X_id == nmodes])
    optimal_modes = np.zeros((nmodes, 3), dtype='float')
    fmtstr = 'Mode ID: {} OptimalBW: {:.2f} Mode: {:.2f} Slope: {}'
    for i in np.arange(nmodes):
        optimal_modes[i] = optimal_mode(dataset, i+1, bw_max)
    return optimal_modes



####################################################################
####################################################################
####################################################################
####################################################################
####################################################################

nmodes = 3
image_filename = 'imageio:camera.png'
image = imageio.imread(image_filename)
distribution = image.ravel()


fig, axes = plt.subplots(figsize=(6,4), dpi=150)

nbins = 256
npoints = 512
kernel = 'gaussian'

x = np.arange(nbins)
y = np.bincount(distribution, minlength=nbins)
y = y/np.sum(y)
hst_xy = (x, y)

estimator = FFTKDE(kernel=kernel, bw='silverman')
kx, ky = estimator.fit(distribution).evaluate(npoints)
ky = ky[(kx>=0) & (kx<=255)] 
kx = kx[(kx>=0) & (kx<=255)] 
kde_xy = (kx, ky)
    
peaks_idx, _ = find_peaks(kde_xy[1])
half = peak_widths(kde_xy[1], peaks_idx, rel_height=0.5)[:2]
peaks = (peaks_idx, half)

estimator = FFTKDE(kernel=kernel, bw='silverman')
kernel_points, kernel_values = estimator.fit(ky).evaluate(npoints)

kx, ky, bw = kernel_points, kernel_values, estimator.bw
bws = bw * np.logspace(1, -1, 101)

mode_lst = find_modeid(distribution, bws)
mode_lst = np.concatenate(mode_lst, axis=0)
dataset = mode_lst

X_bw, X_mode, X_id = dataset[:, 0], dataset[:, 1], dataset[:, 2]
bw_max = np.max(X_bw[X_id == nmodes])
optimal_modes = np.zeros((nmodes, 3), dtype='float')
fmtstr = 'Mode ID: {} OptimalBW: {:.2f} Mode: {:.2f} Slope: {}'
for i in np.arange(nmodes):
    optimal_modes[i] = optimal_mode(dataset, i+1, bw_max)
    print(fmtstr.format(i+1,*(optimal_modes[i])))


#def plot_mode_trace(axes, modes_dataset, minima_points):
#    # Helper variables to make graph calls simpler
#mode, bw, mode_id = modes_dataset[:,1], modes_dataset[:,0], modes_dataset[:,2]
#min_mode, optimal_bw = minima_points[:,1], minima_points[:,0]
#
## How to simplify this custom cmap?
#ncolors = np.max(mode_id)
#cmap = plt.cm.jet
#cmaplist = [cmap(i) for i in range(cmap.N)]
#ncmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
#bounds = np.arange(ncolors)
#norm = mpl.colors.BoundaryNorm(bounds, ncmap.N)
#
#axes.set_yscale('log')
#trc_sct = axes.scatter(mode, bw, s=1, c=mode_id, cmap=ncmap, norm=norm)
#min_sct = axes.scatter(min_mode, optimal_bw, c='k', alpha=.6)
#
#axes.set_ylabel('Bandwidth')
#axes.set_xlabel('Modes')
#axes.set_ylim(np.min(bw), np.max(bw))
#axes.set_xlim(0, 255)
#

