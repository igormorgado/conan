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
        delta = np.inf if len(mode_new) == 1 else np.min(np.diff(mode_new))/2
        d = np.abs(mode_new[:, None] - mode)
        i = np.argmin(d, axis=1)
        g = np.zeros_like(mode_new)
        g[i] = ids[0:d.shape[1]]
        k = (g==0)
        g[k] = np.arange(np.sum(np.logical_not(k))+1, len(g)+1)
        ids = np.copy(g)
        m = np.copy(mode_new)

        tmp_array = np.array([[bw]*g.shape[0], mode_new, g]).T
        dataframe.append(tmp_array)

    return dataframe


mu = np.array([23, 60, 130, 190])
sigma = np.array([8, 13, 15, 19])
p = np.array([.18, .2, .24, .28])
n = 10000
x = rmix(n, mu, sigma, p)

kx, ky, bw = density(x)
bws = bw * 10**np.linspace(1, -1, 101)

mode_lst = find_modeid(x, bws)

