# Place your function here
import numpy as np
import numba as nb


# MC integration setup
setup = {
    'xlow': np.array([0], dtype=np.float64),
    'xupp': np.array([2], dtype=np.float64),
    'ncalls': 10000
}


# test function with numba
@nb.njit(nb.float64(nb.float64[:]))
def MC_INTEGRAND(x):
    """Le page test function"""
    return np.exp(-x[0]) / (1 + (x[0] - 1) ** 2)
