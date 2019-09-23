import numpy as np
from KDEpy import FFTKDE


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
    pass

def func(dataset, idd, bwmax):
    y =  dataset[(dataset[:,2] == idd) & (dataset[:,0] <= bwmax)]
    return y #minslope(y[:,0], y[:,1])

def plot_mode_trace(mode_lst):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Create a new colormab based on number of ids
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    ncmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(ncolors)
    norm = mpl.colors.BoundaryNorm(bounds, ncmap.N)

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    sc = ax.scatter(mode_lst[:,1], mode_lst[:,0], c=mode_lst[:,2], s=1, cmap=ncmap, norm=norm)

    return fig


mu = np.array([23, 60, 130, 190])
sigma = np.array([8, 13, 15, 19])
p = np.array([.18, .2, .24, .28])
n = 10000
x = rmix(n, mu, sigma, p)

# Initial bw estimator based on Silverman
kx, ky, bw = density(x)
bws = bw * 10**np.linspace(1, -1, 101)

mode_lst = find_modeid(x, bws)
mode_lst = np.concatenate(mode_lst, axis=0)


nmodes = 4
bwmax = np.max(mode_lst[mode_lst[:,2] == nmodes][:,0])


# MIN SLOPE
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize

a = func(mode_lst, 4, bwmax)        # CASO PARTICULAR

xx = a[:,0]
yy = a[:,1]

order = np.argsort(xx)
e = (np.max(xx) - np.min(xx)) * 1e-4
f = splrep(xx[order], yy[order])
df2 = lambda xx: ((splev(xx+e, f) - splev(xx-e, f))/(2*e))**2
x0 = np.min(xx)
x0 = np.max(xx)
x0 = 6
# v <- optimize(df2, c(min(x),max(x)))
v = minimize(df2, 6) # , np.max(xx)]))

plt.close('all')
plt.plot(xx,yy, label='trace')
plt.plot(xx,df2(xx),label='d2')
plt.scatter(x0,splev(x0,f), label='x0')
plt.scatter(v.x,splev(v.x,f), label='minima')
plt.scatter(np.max(xx),yy[np.argmax(xx)])
plt.scatter(np.min(xx),yy[np.argmin(xx)])
plt.legend()


