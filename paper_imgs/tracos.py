import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import splrep, splev
from scipy.optimize import fminbound
from scipy.signal import find_peaks, peak_widths
from KDEpy import FFTKDE


def random_modes_distribution(n, mu, sigma, p):
    return np.random.normal(mu, sigma, size=(int(n),1)).ravel()


def build_image_sample(m=512, n=512):
    return random_modes_distribution(m*n,
                                     np.array([25, 60, 130, 190]),
                                     np.array([8, 13, 15, 19]),
                                     np.array([.18, .2, .24, .28]))

def find_modes(kernel_points, kernel_values):
    return kernel_points[np.where((np.append(kernel_values[1:], -np.inf) <= kernel_values) & 
                        (kernel_values >= np.append(-np.inf, kernel_values[:-1])))]


def density(values, bw='silverman', npoints=512):
    estimator = FFTKDE(kernel='gaussian', bw=bw)
    kx, ky = estimator.fit(values).evaluate(npoints)
    ky = ky[(kx>=0) & (kx<=255)] 
    kx = kx[(kx>=0) & (kx<=255)] 
    return kx, ky, estimator.bw


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
    return np.concatenate(dataframe, axis=0)

# REVER ESTA FUNCAO
def min_slope(x, y):
    order = np.argsort(x)
    f = splrep(x[order], y[order])
    e = (np.max(x) - np.min(x)) * 1e-4
    def df2(x, f, e):
        return ((splev(x+e, f) - splev(x-e, f))/(2*e))**2
    bw, slope, err, iters = fminbound(df2, np.min(x), np.max(x), args=(f, e), full_output=True)
    mode = splev(bw, f)
    return bw, mode, slope

def optimal_mode(dataset, modeid, max_bandwidth):
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



nmodes = 3
max_modes = 20
image_filename = 'imageio:camera.png'
image = imageio.imread(image_filename)
distribution = image.ravel()


kx, ky, bw = density(distribution)
kde_xy = (kx, ky)
bws = bw * np.logspace(1, -1, 101)

mode_lst = find_modeid(distribution, bws)
X_bw, X_mode, X_id = mode_lst[:, 0], mode_lst[:, 1], mode_lst[:, 2]
bw_max = np.max(X_bw[X_id == nmodes])
optimal_modes = np.zeros((nmodes, 3), dtype='float')
fmtstr = 'Mode ID: {} OptimalBW: {:.2f} Mode: {:.2f} Slope: {}'
for i in np.arange(nmodes):
    optimal_modes[i] = optimal_mode(mode_lst, i+1, bw_max)
    print(fmtstr.format(i+1,*(optimal_modes[i])))

plotmode_lst = mode_lst[mode_lst[:,2] <= max_modes]
mode, mode_bw, mode_id = plotmode_lst[:,1], plotmode_lst[:,0], plotmode_lst[:,2]
min_mode, optimal_bw = optimal_modes[:,1], optimal_modes[:,0]

plt.close('all')
fig, axes = plt.subplots(figsize=(6,4), dpi=150)
axes.set_yscale('log')
# trc_sct = axes.scatter(mode, mode_bw, s=1, c=mode_id, cmap=ncmap, norm=norm)
trc_sct = axes.scatter(mode, mode_bw, s=1, c='#444444')
min_sct = axes.scatter(min_mode, optimal_bw, c='k', alpha=.6)

axes.set_ylabel(r'Banda ($h$)')
axes.set_xlabel('Modas')
axes.set_ylim(np.min(bw), 100)
axes.set_xlim(0, 255)

axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.spines['left'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_facecolor('#eeeeee')

axes.yaxis.set_label_position('left')
axes.yaxis.set_ticks_position('left')

axes.xaxis.set_visible(True)
axes.xaxis.set_major_locator(MultipleLocator(50))
axes.xaxis.set_minor_locator(MultipleLocator(25))

axes.tick_params(direction='out', color='#000000')
axes.set_axisbelow(True)

axes.grid(which='both')
axes.grid(which='major', color='#fefefe', linewidth=1)
axes.grid(which='minor', color='#fefefe', linewidth=.5)
    
plt.suptitle('Traço de moda: imageio:cameraman.png')
plt.savefig('cameraman_mode_trace.png', dpi=300)
