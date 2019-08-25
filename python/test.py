#!/usr/env python
import time
import numpy as np
import numba as nb


# test function with numba
@nb.njit(nb.float64(nb.float64[:]))
def func(x):
    return np.exp(-x[0]) / (1 + (x[0] - 1) ** 2)


# plain MC implementation
@nb.njit(nb.types.UniTuple(nb.float64,3)(nb.typeof(func),nb.int64,nb.float64[:],nb.float64[:],nb.int64))
def plain(func, dim, xl, xu, calls):
    """A trivial plain MC integrator"""
    vol = 1
    m = 0
    q = 0
    for i in range(dim):
        vol *= xu[i] - xl[i]

    x = np.zeros(dim)
    for n in range(calls):
        for i in range(dim):
            x[i] = xl[i] + np.random.uniform(0, 1) * (xu[i] - xl[i])
        val = func(x)
        d = val - m
        m += d/ (n + 1)
        q += d * d * (n / (n + 1))

    integral = vol * m
    variance = vol * np.sqrt( q / (calls * (calls - 1)))
    error = np.sqrt(variance/calls)
    return integral, variance, error


def MC(func, dim, xl, xu, calls, algorithm):
    """Monte Carlo integrator driver.

    Parameters:
        func (function): the integrand
        dim (int): the space dimension
        xl (list): the lower integration point per dimension
        xu (list): the upper integration point per dimension
        calls (int): number of calls for the MC
        algorithm (str): the algorithm
            - py_plain: plain python MC #TODO: MISER, VEGAS
            - plain: plain numba MC # TODO: parallel, cuda, roc
            - TODO: miser, vegas, + C

    Returns:
        dict: integration result and error
    """
    # some trivial checks
    if dim < 1:
        raise ValueError('dim must be > 0')
    for i in range(dim):
        if xu[i] <= xl[i]:
            raise ValueError('xu must be > xl')

    if algorithm == 'py_plain':
        return plain.py_func(func, dim, xl, xu, calls)
    elif algorithm == 'plain':
        return plain(func, dim, xl, xu, calls)
    else:
        raise Exception()


if __name__ == '__main__':
    """Testing a basic integration"""
    xlow = np.array([0], dtype=np.float64)
    xupp = np.array([2], dtype=np.float64)
    ncalls = 10000

    print('Plain MC pure python:')
    start = time.time()
    r = MC(func, dim=1, xl=xlow, xu=xupp, calls=ncalls, algorithm='py_plain')
    end = time.time()
    print(r)
    print(f'time (s): {end-start}\n')

    print('Plain MC numba:')
    start = time.time()
    r = MC(func, dim=1, xl=xlow, xu=xupp, calls=ncalls, algorithm='plain')
    end = time.time()
    print(r)
    print(f'time (s): {end-start}\n')
