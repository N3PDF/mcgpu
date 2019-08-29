# Place your function here
import numpy as np
import numba as nb

DIM = 2

# MC integration setup
setup = {
    'xlow': np.array([0]*DIM, dtype=np.float64),
    'xupp': np.array([1]*DIM, dtype=np.float64),
    'ncalls': int(1e6),
    'dim': DIM
}


# test function with numba
@nb.njit(nb.float64(nb.float64[:]))
def MC_INTEGRAND(xarr):
    """Le page test function"""
    a = 0.1
    n = DIM
    n100 = 100*n
    pref = pow(1.0/a/np.sqrt(np.pi), n)
    coef = 0
    for i in range(n100+1):
        coef += i
    for x in xarr:
        coef += pow( (x-1.0/2.0)/a, 2 )
    coef -= (n100+1)*n100/2.0

    return pref*np.exp(-coef)
