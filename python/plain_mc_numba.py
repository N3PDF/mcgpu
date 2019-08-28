#!/usr/env python
import time
from integrand import MC_INTEGRAND, setup
import numpy as np
import numba as nb


# plain MC implementation
@nb.njit(nb.types.UniTuple(nb.float64,3)(nb.int64,nb.float64[:],nb.float64[:],nb.int64))
def plain(dim, xl, xu, calls):
    """A trivial plain MC integrator"""

    for i in range(dim):
        if xu[i] <= xl[i]:
            raise ValueError('xu must be greater than xl')
        if xu[i] - xl[i] > 1e308:
            raise ValueError('are you really sure about the xu and xl?')

    vol = 1
    m = 0
    q = 0
    for i in range(dim):
        vol *= xu[i] - xl[i]

    x = np.zeros(dim)
    for n in range(calls):
        for i in range(dim):
            x[i] = xl[i] + np.random.uniform(0, 1) * (xu[i] - xl[i])
        val = MC_INTEGRAND(x)
        d = val - m
        m += d / (n + 1)
        q += d * d * (n / (n + 1))

    integral = vol * m
    variance = vol * np.sqrt(q / (calls * (calls - 1)))
    error = np.sqrt(variance/calls)
    return integral, variance, error


if __name__ == '__main__':
    """Testing a basic integration"""
    ncalls = setup['ncalls']
    xlow = setup['xlow']
    xupp = setup['xupp']

    print(f'Plain MC pure python, ncalls={ncalls}:')
    start = time.time()
    r = plain(dim=1, xl=xlow, xu=xupp, calls=ncalls)
    end = time.time()
    print(r)
    print(f'time (s): {end-start}')
