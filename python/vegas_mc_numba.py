#!/usr/env python
import time
from integrand import MC_INTEGRAND, setup
import numpy as np
import numba as nb

BINS_MAX = 50


@nb.njit(nb.float64(nb.float64[:],nb.int64[:],nb.int64,nb.int64[:],nb.int64,nb.float64[:],nb.float64[:],nb.float64[:,:]))
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


@nb.njit(nb.types.Tuple((nb.float64,nb.int64))(nb.int64,nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:,:]))
def initialize_grid(dim, xl, xu, sdelx, sxi):
    """Initialize grid"""
    vol = 1
    sbins = 1
    for j in range(dim):
        dx = xu[j] - xl[j]
        sdelx[j] = dx
        vol *= dx
        sxi[0,j] = 0
        sxi[1,j] = 1
    svol = vol
    return svol, sbins


@nb.njit(nb.int64(nb.int64,nb.int64,nb.int64,nb.float64[:,:],nb.float64[:]))
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


@nb.njit(nb.void(nb.int64,nb.int64,nb.float64[:,:]))
def reset_grid_values(dim, bins, sd):
    for i in range(bins):
        for j in range(dim):
            sd[i, j] = 0.0


@nb.njit(nb.void(nb.int64,nb.int64[:]))
def init_box_coord(dim, box):
    for i in range(dim):
        box[i] = 0


@nb.njit(nb.void(nb.int64,nb.float64[:,:],nb.int64[:],nb.float64))
def accumulate_distribution(dim, sd, sbin, y):
    for j in range(dim):
        i = sbin[j]
        sd[i, j] += y


@nb.njit(nb.int64(nb.int64,nb.int64[:],nb.int64))
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
@nb.njit(
    nb.types.UniTuple(nb.float64,4)(
        nb.int64,nb.float64[:],nb.float64[:],nb.int64,nb.int64,
        nb.int64,nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64,
        nb.float64[:],nb.float64[:,:],nb.int64[:],nb.int64[:],nb.float64,
        nb.int64,nb.float64,nb.int64,nb.float64,nb.float64,nb.float64
        )
    )
def vegas(dim, xl, xu, calls, stage, sbins, sdelx,
          sxi, sxin, svol, sx, sd, sbox, sbin, sjac,
          sboxes, ssum_wgts, ssamples, swtd_int_sum,
          schi_sum, schisq):

    for i in range(dim):
        if xu[i] <= xl[i]:
            raise ValueError('xu must be greater than xl')
        if xu[i] - xl[i] > 1e308:
            raise ValueError('are you really sure about the xu and xl?')

    if stage == 0:
        svol, sbins = initialize_grid(dim, xl, xu, sdelx, sxi)

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
        intgrl_sq = 0
        tss = 0

        reset_grid_values(dim, sbins, sd)
        init_box_coord(dim, sbox)

        while True:
            m = 0
            q = 0
            f_sq_sum = 0.0

            for k in range(calls):

                bin_vol = random_point(sx, sbin, dim, sbox, sbins, xl, sdelx, sxi)

                fval = sjac * bin_vol * MC_INTEGRAND(sx)

                d = fval - m
                m += d / (k + 1.0)
                q += d * d * (k / (k + 1.0))

                f_sq = fval * fval
                accumulate_distribution(dim, sd, sbin, f_sq)

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

        intgrl_sq = intgrl * intgrl
        sig = np.sqrt(var)

        if wgt > 0.0:
            ssamples += 1
            ssum_wgts += wgt
            swtd_int_sum += intgrl * wgt
            schi_sum += intgrl_sq * wgt

            cum_int = swtd_int_sum / ssum_wgts
            cum_sig = np.sqrt(1 / ssum_wgts)

            if ssamples == 1:
                schisq = 0
            else:
                m = 0
                if ssum_wgts > 0:
                    m = swtd_int_sum / ssum_wgts
                q = intgrl - m
                schisq *= (ssamples - 2.0)
                schisq += (wgt / (1 + (wgt / ssum_wgts))) * q * q
                schisq /= (ssamples - 1.0)
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
        self.schi_sum = 0
        self.schisq = 0

    def integrate(self, xl, xu, calls, stage):
        r = vegas(self.dim, xl, xu, calls, stage,
                self.sbins, self.sdelx, self.sxi, self.sxin, self.svol,
                self.sx, self.sd, self.sbox, self.sbin, self.sjac, self.sboxes,
                self.ssum_wgts, self.ssamples, self.swtd_int_sum, self.schi_sum, self.schisq)
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
