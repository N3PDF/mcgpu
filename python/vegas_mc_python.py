#!/usr/env python
import time
from integrand import MC_INTEGRAND, setup
import numpy as np
import numba as nb

BINS_MAX = 50


def random_point(x, sbin, dim, box, bins, xl, delx, sxi):
    vol = 1
    for i in range(dim):
        z = (box[i] + np.random.uniform(0,1)) * bins
        k = nb.int64(z)
        sbin[i] = k
        if z == 0:
            bin_width = sxi[1, i]
            y = z * bin_width
        else:
            bin_width = sxi[k+1, i] - sxi[k, i]
            y = sxi[k, i] + (z - k) * bin_width
        x[i] = xl[i] + y * delx[i]
        vol *= bin_width
    return vol


def resize_grid(dim, sbins, bins, sxi, sxin):
    """Resize grid maintaining the density distribution"""
    pts_per_bin = sbins / bins
    for j in range(dim):
        i = 1
        xnew = 0
        dw = 0

        for k in range(1, sbins+1):
            dw += 1.0
            xold = xnew
            xnew = sxi[k, j]

            while dw > pts_per_bin:
                dw -= pts_per_bin
                sxin[i] = xnew - (xnew - xold) * dw
                i += 1

        for k in range(1, bins):
            sxi[k, j] = sxin[k]

        sxi[bins, j] = 1

    return bins


def change_box_coord(dim, box, boxes):
    j = dim - 1
    ng = boxes
    while j >= 0:
        box[j] = (box[j] + 1) % ng
        if box[j] != 0:
            return 1
        j -= 1
    return 0


# vegas MC implementation
def vegas(dim, xl, xu, calls, stage, sbins, sdelx,
          sxi, sxin, svol, sx, sd, sbox, sbin, sjac,
          sboxes, ssum_wgts, ssamples, swtd_int_sum):

    for i in range(dim):
        if xu[i] <= xl[i]:
            raise ValueError('xu must be greater than xl')
        if xu[i] - xl[i] > 1e308:
            raise ValueError('are you really sure about the xu and xl?')

    if stage == 0:
        vol = 1
        sbins = 1
        for j in range(dim):
            dx = xu[j] - xl[j]
            sdelx[j] = dx
            vol *= dx
            sxi[0,j] = 0
            sxi[1,j] = 1
        svol = vol

    if stage == 1:
        ssum_wgts = 0
        ssamples = 0

    if stage <= 2:
        bins = BINS_MAX
        sboxes = 1
        sjac = svol * bins ** dim / calls

        if bins != sbins:
            sbins = resize_grid(dim, sbins, bins, sxi, sxin)

    cum_int = 0.0
    cum_sig = 0.0

    iterations = 5
    for it in range(iterations):
        intgrl = 0
        tss = 0

        for i in range(sbins):
            for j in range(dim):
                sd[i, j] = 0.0

        for i in range(dim):
            sbox[i] = 0

        while True:
            m = 0
            q = 0
            f_sq_sum = 0.0

            for k in range(calls):

                bin_vol = random_point(sx, sbin, dim, sbox, sbins, xl, sdelx, sxi)

                fval = sjac * bin_vol * MC_INTEGRAND.py_func(sx)

                d = fval - m
                m += d / (k + 1.0)
                q += d * d * (k / (k + 1.0))

                f_sq = fval * fval

                # accumulate distribution
                for j in range(dim):
                    i = sbin[j]
                    sd[i, j] += f_sq

            intgrl += m * calls
            f_sq_sum = q * calls
            tss += f_sq_sum

            if not change_box_coord(dim, sbox, sboxes):
                break

        var = tss / (calls - 1)

        if var > 0:
            wgt = 1.0 / var
        elif ssum_wgts > 0:
            wgt = ssum_wgts / ssamples
        else:
            wgt = 0.0

        if wgt > 0.0:
            ssum_wgts += wgt
            swtd_int_sum += intgrl * wgt
            cum_int = swtd_int_sum / ssum_wgts
            cum_sig = np.sqrt(1 / ssum_wgts)
        else:
            cum_int += (intgrl - cum_int) / (it + 1.0)
            cum_sig = 0.0

    return svol, cum_int, cum_sig, np.sqrt(cum_sig/calls)


class make_vegas:
    """A Vegas MC integrator using importance sampling"""
    def __init__(self, dim):
        self.dim = dim
        self.sbins = 0
        self.sbin = np.zeros(dim, dtype=np.int64)
        self.sdelx = np.zeros(dim)
        self.sxi = np.zeros(shape=(BINS_MAX+1, dim))
        self.sxin = np.zeros(shape=(BINS_MAX+1))
        self.svol = 0
        self.sx = np.zeros(dim)
        self.sd = np.zeros(shape=(BINS_MAX, dim))
        self.sbox = np.zeros(dim, dtype=np.int64)
        self.sjac = 0
        self.sboxes = 0
        self.ssum_wgts = 0
        self.ssamples = 0
        self.swtd_int_sum = 0

    def integrate(self, xl, xu, calls, stage):
        r = vegas(self.dim, xl, xu, calls, stage,
                self.sbins, self.sdelx, self.sxi, self.sxin, self.svol,
                self.sx, self.sd, self.sbox, self.sbin, self.sjac, self.sboxes,
                self.ssum_wgts, self.ssamples, self.swtd_int_sum)
        self.svol = r[0]
        return r[1:]


if __name__ == '__main__':
    """Testing a basic integration"""
    ncalls = setup['ncalls']
    xlow = setup['xlow']
    xupp = setup['xupp']
    dim = setup['dim']

    print(f'VEGAS MC stage=0 numba, ncalls={ncalls}:')
    start = time.time()
    v = make_vegas(dim=dim)
    r = v.integrate(xl=xlow, xu=xupp, calls=ncalls, stage=0)
    end = time.time()
    print(r)
    print(f'time (s): {end-start}')

    print(f'VEGAS MC stage=1 numba, ncalls={ncalls}:')
    start = time.time()
    r = v.integrate(xl=xlow, xu=xupp, calls=ncalls, stage=1)
    end = time.time()
    print(r)
    print(f'time (s): {end-start}')
