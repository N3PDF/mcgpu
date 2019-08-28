# Place your function here
import numpy as np
import numba as nb

DIM = 1

# MC integration setup
setup = {
    'xlow': np.array([0]*DIM, dtype=np.float64),
    'xupp': np.array([1]*DIM, dtype=np.float64),
    'ncalls': 10000,
    'dim': DIM
}


# test function with numba
@nb.njit(nb.float64(nb.float64[:]))
def MC_INTEGRAND(x):
    """Le page test function"""
    a = 0.1
    n = len(x)
    pref = (1.0/a/np.sqrt(np.pi))**n
    coef = 0
    for i in range(1, 100*n+1):
        coef = coef + np.float64(i)
    for i in range(n):
        coef = coef + (x[i] - 1.0/2.0)**2/a**2
    coef = coef - np.float64(100*n)*np.float64(100*n+1)/2.0

    return pref*np.exp(-coef)