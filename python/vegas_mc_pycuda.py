#!/usr/env python
import time
from integrand import MC_INTEGRAND, setup
import numpy as np
import numba as nb
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

BINS_MAX = 50
ALPHA = 1.5

@nb.njit(nb.float64())
def internal_rand():
    """ Generates a random number """
    return np.random.uniform(0,1)

@nb.njit(nb.float64[:,:](nb.int64, nb.int64, nb.float64[:], nb.int64[:]))
def loop(n_events, n_dim,  fres2, all_div_indexes):
    arr_res2 = np.zeros((n_dim, BINS_MAX))
    for i in range(n_events):
        for j in range(n_dim):
            arr_res2[j, all_div_indexes[i*n_dim + j]] += fres2[i]
    return arr_res2


@nb.njit(nb.float64[:](nb.int64, nb.int64, nb.float64[:,:], nb.float64[:], nb.int64[:]))
def generate_random_array(n_events, n_dim, divisions, x, div_index):
    """
    Generates a random array
    # Arguments in:
        - n_dim: number of dimensions
        - divisions: an array defining the grid divisions

    # Arguments out:
        - x: array of dimension n_dim with the random number
        - div_index: array of dimension n_dim with the subdivision
                     of each point

    # Returns:
        - wgt: weight of the point
    """
    all_wgts = np.zeros(n_events)
    for j in range(n_events):
        reg_i = 0.0
        reg_f = 1.0
        wgt = 1.0
        index = j*n_dim
        for i in range(n_dim):
            rn = internal_rand()
            # Get a random number randomly assigned to a subdivision
            xn = BINS_MAX*(1.0 - rn)
            int_xn = max(0, min(int(xn), BINS_MAX))
            # In practice int_xn = int(xn)-1 unless xn < 1
            aux_rand = xn - int_xn
            if int_xn == 0:
                x_ini = 0.0
            else:
                x_ini = divisions[i, int_xn - 1]
            xdelta = divisions[i, int_xn] - x_ini
            rand_x = x_ini + xdelta*aux_rand
            x[i+index] = reg_i + rand_x*(reg_f - reg_i)
            wgt *= xdelta*BINS_MAX
            div_index[i+index] = int_xn
        all_wgts[j] = wgt
    return all_wgts

@nb.njit(nb.void(nb.float64[:], nb.float64, nb.float64[:]))
def rebin(rw, rc, subdivisions):
    """ broken from function above to use it for initialiation """
    k = -1
    dr = 0.0
    aux = []
    for i in range(BINS_MAX-1):
        old_xi = 0.0
        while rc > dr:
            k += 1
            dr += rw[k]
        if k > 0:
            old_xi = subdivisions[k-1]
        old_xf = subdivisions[k]
        dr -= rc
        delta_x = old_xf-old_xi
        aux.append(old_xf - delta_x*(dr / rw[k]))
    aux.append(1.0)
    for i, tmp in enumerate(aux):
        subdivisions[i] = tmp

@nb.njit(nb.void(nb.int64, nb.float64[:], nb.float64, nb.float64[:,:]))
def rebin2(index, rw, rc, subdivisions):
    """ broken from function above to use it for initialiation """
    k = -1
    dr = 0.0
    aux = []
    for i in range(BINS_MAX-1):
        old_xi = 0.0
        while rc > dr:
            k += 1
            dr += rw[k]
        if k > 0:
            old_xi = subdivisions[index, k-1]
        old_xf = subdivisions[index, k]
        dr -= rc
        delta_x = old_xf-old_xi
        aux.append(old_xf - delta_x*(dr / rw[k]))
    aux.append(1.0)
    for i, tmp in enumerate(aux):
        subdivisions[index, i] = tmp


@nb.njit(nb.void(nb.int64, nb.float64[:], nb.float64[:,:]))
def refine_grid(j, res_sq, subdivisions):
    """
        Resize the grid
    # Arguments in:
        - res_sq: array with the accumulative sum for each division for one dim
    # Arguments inout:
        - subdivisions: the array the defining the vegas grid divisions for one dim
    """
    index = j * BINS_MAX
    # First we smear out the array div_sq, where we have store
    # the value of f^2 for each sub_division for each dimension
    aux = [
            (res_sq[0 + index] + res_sq[1 + index])/2.0
            ]
    for i in range(1, BINS_MAX-1):
        tmp = (res_sq[i-1 + index] + res_sq[i + index] + res_sq[i+1+index])/3.0
        if tmp < 1e-30:
            tmp = 1e-30
        aux.append(tmp)
    tmp = (res_sq[BINS_MAX-2+index] + res_sq[BINS_MAX-1+index])/2.0
    aux.append(tmp)
    aux_sum = np.sum(np.array(aux))
    # Now we refine the grid according to
    # journal of comp phys, 27, 192-203 (1978) G.P. Lepage
    rw = []
    for res in aux:
        tmp = pow( (1.0 - res/aux_sum)/(np.log(aux_sum) - np.log(res)), ALPHA )
        rw.append(tmp)
    rw = np.array(rw)
    rc = np.sum(rw)/BINS_MAX
    rebin2(index, rw, rc, subdivisions)


def vegas(n_dim, n_iter, n_events, results, sigmas):
    """
    # Arguments in:
        n_dim: number of dimensions
        n_iter: number of iterations
        n_events: number of events per iteration

    # Arguments out:
        results: array with all results by iteration
        sigmas: array with all errors by iteration

    # Returns:
        - integral value
        - error
    """
    # Initialize variables
    xjac = 1.0/n_events
    divisions = np.zeros( (n_dim, BINS_MAX) )
    divisions[:, 0] = 1.0
    # Do a fake initialization at the begining
    rw_tmp = np.ones(BINS_MAX)
    for i in range(n_dim):
        rebin(rw_tmp, 1.0/BINS_MAX, divisions[i])

    # "allocate" arrays
    x = np.zeros(n_dim)
    div_index = np.zeros(n_dim, dtype = np.int64)
    all_results = []

    mod = SourceModule("""
    __global__
    void events_kernel(double *all_randoms, double *all_xwgts, int n, int n_events, double xjac, double *all_res, double *all_res2) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int block_size = blockDim.x;

    int index = block_id*block_size + thread_id;
    int grid_dim = gridDim.x;
    int stride = block_size * grid_dim;

    // shared lepage
    double a = 0.1;
    double pref = pow(1.0/a/sqrt(M_PI), n);

    for (int i = index; i < n_events; i += stride) {
        double wgt = all_xwgts[i]*xjac;

        double coef = 0.0;
        for (int j = 0; j < n; j++) {
            coef += pow( (all_randoms[i*n + j] - 1.0/2.0)/a, 2 );
        }
        double lepage = pref*exp(-coef);

        double tmp = wgt*lepage;
        all_res[i] = tmp;
        all_res2[i] = pow(tmp,2);
    }
    }
    """)

    events_kernel = mod.get_function("events_kernel")

    # Loop of iterations
    for k in range(n_iter):
        res = 0.0
        res2 = 0.0
        sigma = 0.0


        NN = n_dim*n_events
        all_randoms = np.empty(NN)
        all_xwgts = np.empty(n_events)

        all_div_indexes = np.zeros(NN, dtype=np.int64)
        #for i in range(n_events):
        #    all_xwgts[i] = generate_random_array(i, n_dim, divisions, all_randoms, all_div_indexes)
        all_xwgts = generate_random_array(n_events, n_dim, divisions, all_randoms, all_div_indexes)

        #all_randoms = gpuarray.to_gpu(all_randoms)
        #all_xwgts = gpuarray.to_gpu(all_xwgts)

        #from pycuda import driver
        #all_res = gpuarray.to_gpu(np.empty(n_events, dtype=np.float64))
        #all_res2 = gpuarray.to_gpu(np.empty(n_events, dtype=np.float64))
        #dummy = np.zeros(n_events, dtype=np.int32)
        #all_res = driver.managed_empty(dummy.shape, dummy.dtype, "C", 0)
        #all_res2 = driver.managed_zeros_like(np.zeros(n_events, dtype=np.float64))
        threads = 256
        blocks = int((n_events + threads - 1) / threads)
        #events_kernel(all_randoms, all_xwgts, np.int32(n_dim), np.int32(n_events), np.float64(xjac), all_res, all_res2, block=(1,1,1))

        fres = np.zeros(n_events, dtype=np.float64)
        fres2 = np.zeros(n_events, dtype=np.float64)
        events_kernel(drv.In(all_randoms), drv.In(all_xwgts), np.int32(n_dim), np.int32(n_events), np.float64(xjac), drv.Out(fres),
                drv.Out(fres2), block=(threads,1, 1), grid = (blocks, 1))

        #fres = all_res.get()
        #fres2 = all_res2.get()
        res = np.sum(fres)
        res2 = np.sum(fres2)

        arr_res2 = loop(n_events, n_dim, fres2, all_div_indexes)
        arr_res2 = arr_res2.flatten()

        err_tmp2 = max((n_events*res2 - res*res), 1e-30)
        sigma = np.sqrt(err_tmp2)
        print("Results for interation {0}: {1} +/- {2}".format(k+1, res, sigma))
        results[k] = res
        sigmas[k] = sigma
        all_results.append( (res, sigma) )
        for j in range(n_dim):
            refine_grid(j, arr_res2, divisions)


    # Compute the final results
    aux_res = 0.0
    weight_sum = 0.0
    for result in all_results:
        res = result[0]
        sigma = result[1]
        wgt_tmp = 1.0/pow(sigma, 2)
        aux_res += res*wgt_tmp
        weight_sum += wgt_tmp

    final_result = aux_res/weight_sum
    sigma = np.sqrt(1.0/weight_sum)
    return final_result, sigma

class make_vegas:
    """A Vegas MC integrator using importance sampling"""
    def __init__(self, dim, xl = None, xu = None):
        self.dim = dim
        # At the moment we save xl, xu but it is not used
        self.xl = xl
        self.xu = xu
#         self.sbins = 0
#         self.sbin = np.zeros(dim, dtype=np.int64)
#         self.sdelx = np.zeros(dim)
#         self.sxi = np.zeros(shape=(BINS_MAX+1, dim))
#         self.sxin = np.zeros(shape=(BINS_MAX+1))
#         self.svol = 0
#         self.sd = np.zeros(shape=(BINS_MAX, dim))
#         self.sbox = np.zeros(dim, dtype=np.int64)
#         self.sjac = 0
#         self.ssum_wgts = 0
#         self.ssamples = 0
#         self.swtd_int_sum = 0

    def integrate(self, iters = 5, calls = 1e4):
        results = np.zeros(iters)
        sigmas = np.zeros(iters)
        r = vegas(self.dim, iters, calls, results, sigmas)
        for k, (res, sigma) in enumerate(zip(results, sigmas)):
            print("Results for interation {0}: {1} +/- {2}".format(k+1, res, sigma))
        return r
#         r = vegas(self.dim, xl, xu, calls, stage,
#                 self.sbins, self.sdelx, self.sxi, self.sxin, self.svol,
#                 self.sd, self.sbox, self.sbin, self.sjac,
#                 self.ssum_wgts, self.ssamples, self.swtd_int_sum)
#         self.svol = r[0]
#         return r[1:]


if __name__ == '__main__':
    """Testing a basic integration"""
    ncalls = setup['ncalls']
    xlow = setup['xlow']
    xupp = setup['xupp']
    dim = setup['dim']

    print(f'VEGAS MC numba, ncalls={ncalls}:')
    start = time.time()
    v = make_vegas(dim=dim, xl = xlow, xu = xupp)
    r = v.integrate(calls=ncalls)
    end = time.time()
    print(r)
    print(f'time (s): {end-start}')
