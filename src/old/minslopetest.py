X = mode_lst
X_bw, X_mode, X_id = X[:,0], X[:,1], X[:,2]

#
# RETAIN DESIRED MODES
#
n_modes = 4
bw_max = np.max(X_bw[X_id == n_modes])

idd = n_modes           # PARTICULAR
Y = X[(X_id == idd) & (X_bw <= bw_max)]
Y_bw, Y_mode, Y_id = Y[:,0], Y[:,1], Y[:,2]

# FUNCAO MINSLOPE(x = Y$bw, y = Y$mode)
MX, MY = Y_bw, Y_mode

# f<-splinefun(x,y)
order = np.argsort(MX)
f = splrep(MX[order], MY[order])

# e <- diff(range(x)) * 1e-4
e = (np.max(MX) - np.min(MX)) * 1e-4

# df2 <- function(MX) ((f(x+e) - f(x-2)) / (2*e))^2
def df2(x, f, e):
    return ((splev(x+e, f) - splev(x-e, f))/(2*e))**2

# v <- optimize(df2, c(min(x), max(x)))
v = fminbound(df2, np.min(MX), np.max(MX), args=(f, e), full_output=True)

