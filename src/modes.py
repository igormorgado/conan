import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.optimize import fminbound
from KDEpy import FFTKDE

plt.style.use('ggplot')

# Equivalent to density in R
def density(x, bw='silverman', n=512):
    estimator = FFTKDE(kernel='gaussian', bw=bw)
    kx, ky = estimator.fit(x).evaluate(n)
    return kx, ky, estimator.bw


def rmix(n, mu, sigma, p):
    return np.random.normal(mu, sigma, size=(int(n),1)).ravel()


def find_modes(kx, ky):
    return kx[np.where((np.append(ky[1:], -np.inf) < ky) & 
                       (ky > np.append(-np.inf, ky[:-1])))]


def find_modeid(data, bws):
    dataframe = []
    ids = np.array([1])
    mode = np.array(np.mean(data))
    for bw in bws:
        kx, ky, _ = density(data, bw=bw)
        mode_new = np.sort(find_modes(kx, ky))
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


def min_slope(x, y):
    order = np.argsort(x)
    f = splrep(x[order], y[order])
    e = (np.max(x) - np.min(x)) * 1e-4
    def df2(x, f, e):
        return ((splev(x+e, f) - splev(x-e, f))/(2*e))**2
    bw, slope, err, iters = fminbound(df2, np.min(x), np.max(x), args=(f, e), full_output=True)
    mode = splev(bw, f)
    return bw, mode, slope


def plot_mode_trace(mode_lst, result):
    # TODO: 
    #   PLOT THE POINTS OF MODES optimal in trace
    #   Interpolate values for a denser plot??
    ncolors = np.max(mode_lst[:,2])
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ncmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(ncolors)
    norm = mpl.colors.BoundaryNorm(bounds, ncmap.N)

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    sc = ax.scatter(mode_lst[:,1], mode_lst[:,0], c=mode_lst[:,2], s=1, cmap=ncmap, norm=norm)
    sc2 = ax.scatter(result[:,1], result[:,0])
    print('x', result[:,1])
    print('y', result[:,0])
    fig.suptitle('Mode trace')

    return fig


def plot_histogram(distribution, modes):
    #TODO: Plot histogram with vlines in modes discovered
    fig, ax = plt.subplots()
    fig.suptitle('Histogram with modes')
    y, bins, patches = ax.hist(distribution, 100, density=True)
    mlines = ax.vlines(modes, 0, np.max(y), 'k', linestyles='dashed')

    return fig


def optimal_mode(dataset, modeid, max_bandwidth):
    """ Given a modeid from dataset and a maximum search bandwidth return
        the optimal bandwidth, mode value and slope
    """
    assert (modeid > 0), "ModeID should be higher than 0"
    y = dataset[(dataset[:,2] == modeid) & (dataset[:,0] <= max_bandwidth)]
    opt_bandwidth, mode, slope = min_slope(y[:,0], y[:,1])
    return opt_bandwidth, mode, slope


#################################################
#
# MAIN
#
#################################################

# Creates synthetic data
mu = np.array([23, 60, 130, 190])
sigma = np.array([8, 13, 15, 19])
p = np.array([.18, .2, .24, .28])
n = 10000
x = rmix(n, mu, sigma, p)

# Initial bw estimator based on Silverman
kx, ky, bw = density(x)
bws = bw * 10**np.linspace(1, -1, 101)

# Extraction of modes and bandwidths
mode_lst = find_modeid(x, bws)
mode_lst = np.concatenate(mode_lst, axis=0)
X = mode_lst
X_bw, X_mode, X_id = X[:,0], X[:,1], X[:,2]

# The code
n_modes = 5 # Because I know there is 4 modes... how to do automatically?
bw_max = np.max(X_bw[X_id == n_modes])

result = np.zeros((n_modes, 3), dtype='float')
for i in np.arange(1, n_modes+1):
    result[i-1] = optimal_mode(X, i, bw_max)
    print('Mode ID: {} OptimalBW: {:.2f} Mode: {:.2f} Slope: {}'.format(i,*(result[i-1])))

plt.close('all')
plot_mode_trace(X, result)
plot_histogram(x, result[:,1])
plt.show()
