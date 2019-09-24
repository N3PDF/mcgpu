#!/usr/env python
import time
from integrand import MC_INTEGRAND, setup
import numpy as np
import numba as nb
import pyopencl as cl
import pyopencl.array as pycl_array

context = cl.create_some_context()
queue = cl.CommandQueue(context)

BINS_MAX = 30
ALPHA = 0.1

kernelA = cl.Program(context, """
__kernel void events_kernel(__global const double *all_randoms, __global const double *all_xwgts, int n, int n_events, double xjac, __global double *all_res, __global double *all_res2) {
    int block_id = get_group_id(0);
    int thread_id = get_local_id(0);
    int block_size = get_local_size(0);

    int index = block_id*block_size + thread_id;
    int grid_dim = get_num_groups(0);
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
""").build()

kernelB = cl.Program(context, """
#define BINS_MAX 30

double bad_rand(int* seed) // 1 <= *seed < m
{
    int const a = 16807; //ie 7**5
    int const m = 2147483647; //ie 2**31-1

    *seed = (int) ((long) (*seed * a))%m;
    double rn = (double) *seed / INT_MAX;
    return (rn + 1.0)/2.0;
}

uint MWC64X(uint2 *state)
{
    enum { A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);              // Pack the state back up
    return res;                       // Return the next result
}

__kernel void generate_random_array_kernel(const int n_events, const int n_dim, __global const double *divisions, __global double *all_randoms, __global double *all_wgts, __global int *all_div_indexes) {
    double reg_i = 0.0;
    double reg_f = 1.0;

    int block_id = get_group_id(0);
    int thread_id = get_local_id(0);
    int block_size = get_local_size(0);

    int index = block_id*block_size + thread_id;
    int grid_dim = get_num_groups(0);
    int stride = block_size * grid_dim;

    // Use curandState_t to keep track of the seed, which is thread dependent
    uint2 state = index;
    int seed = index;

    for (int j = index; j < n_events; j+= stride) {
        double wgt = 1.0;
        for (int i = 0; i < n_dim; i++) {
            //double rn = (double) MWC64X(&state)/UINT_MAX;
            double rn = bad_rand(&seed);
            double xn = BINS_MAX*(1.0 - rn);
            int int_xn = max(0, min( (int) xn, BINS_MAX));
            double aux_rand = xn - int_xn;
            double x_ini = 0.0;
            if (int_xn > 0) {
                x_ini = divisions[BINS_MAX*i + int_xn-1];
            }
            double xdelta = divisions[BINS_MAX*i + int_xn] - x_ini;
            double rand_x = x_ini + xdelta*aux_rand;
            wgt *= xdelta*BINS_MAX;
            // Now we need to add an offset to the arrays
            all_randoms[j*n_dim + i] = reg_i + rand_x*(reg_f - reg_i);
            all_div_indexes[j*n_dim + i] = int_xn;
            }
        all_wgts[j] = wgt;
    }
}""").build()


@nb.njit(nb.float64())
def internal_rand():
    """ Generates a random number """
    return np.random.uniform(0,1)

@nb.njit(nb.float64[:](nb.int64, nb.int64, nb.float64[:], nb.int32[:]))
def loop(n_events, n_dim,  fres2, all_div_indexes):
    arr_res2 = np.zeros(n_dim * BINS_MAX)
    for i in range(n_events):
        for j in range(n_dim):
            arr_res2[j * BINS_MAX + all_div_indexes[i*n_dim + j]] += fres2[i]
    return arr_res2

@nb.njit(nb.void(nb.float64[:], nb.float64, nb.float64[:], nb.int64))
def rebin(rw, rc, subdivisions, index):
    """ broken from function above to use it for initialiation """
    k = -1
    dr = 0.0
    aux = [] #np.zeros(BINS_MAX, dtype=np.float64)
    for i in range(BINS_MAX-1):
        old_xi = 0.0
        while rc > dr:
            k += 1
            dr += rw[k]
        if k > 0:
            old_xi = subdivisions[k-1+index]
        old_xf = subdivisions[k+index]
        dr -= rc
        delta_x = old_xf-old_xi
        aux.append(old_xf - delta_x*(dr / rw[k]))
    aux.append(1.0)
    for i, tmp in enumerate(aux):
        subdivisions[i+index] = tmp

@nb.njit(nb.void(nb.int64, nb.float64[:], nb.float64[:]))
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
    rebin(rw, rc, subdivisions, index)

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
    divisions = np.zeros(n_dim * BINS_MAX, dtype=np.float64)
    for j in range(n_dim):
        divisions[j * BINS_MAX] = 1.0

    # Do a fake initialization at the begining
    rw_tmp = np.ones(BINS_MAX, dtype=np.float64)
    for j in range(n_dim):
        rebin(rw_tmp, 1.0/BINS_MAX, divisions, j * BINS_MAX)

    total_res = 0.0
    total_weight = 0.0

    # threads and blocks
    threads = 1000
    grid_size = int((n_events + threads - 1)/threads)*threads

    # Loop of iterations
    for k in range(n_iter):
        res = 0.0
        res2 = 0.0
        sigma = 0.0

        # input arrays
        NN = n_dim*n_events
        all_randoms = np.empty(NN, dtype=np.float64)
        all_div_indexes = np.zeros(NN, dtype=np.int32)

        # output arrays
        all_res = pycl_array.to_device(queue, np.zeros(n_events))
        all_res2 = pycl_array.to_device(queue, np.zeros(n_events))

        cl_all_randoms = pycl_array.to_device(queue, all_randoms)
        cl_all_xwgts = pycl_array.to_device(queue, np.zeros(n_events))
        cl_all_div_indexes = pycl_array.to_device(queue, all_div_indexes)
        cl_divisions = pycl_array.to_device(queue, divisions)


        kernelB.generate_random_array_kernel(queue, (grid_size,), None,
                                             np.int32(n_events), np.int32(n_dim),
                                             cl_divisions.data, cl_all_randoms.data,
                                             cl_all_xwgts.data, cl_all_div_indexes.data)
        queue.finish()

        kernelA.events_kernel(queue, (grid_size,), None,
                              cl_all_randoms.data, cl_all_xwgts.data, np.int32(n_dim), np.int32(n_events),
                              np.float64(xjac), all_res.data, all_res2.data)

        queue.finish()
        aa = all_res.get()
        bb = all_res2.get()
        res = np.sum(aa)
        res2 = np.sum(bb)

        arr_res2 = loop(n_events, n_dim, all_res2.get(), cl_all_div_indexes.get())

        err_tmp2 = max((n_events*res2 - res*res)/(n_events-1.0), 1e-30)
        sigma = np.sqrt(err_tmp2)
        print("Results for interation {0}: {1} +/- {2}".format(k+1, res, sigma))

        for j in range(n_dim):
            refine_grid(j, arr_res2, divisions)

        wgt_tmp = 1.0/pow(sigma, 2)
        total_res += res * wgt_tmp
        total_weight += wgt_tmp

    final_result = total_res/total_weight
    sigma = np.sqrt(1.0/total_weight)
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
        #for k, (res, sigma) in enumerate(zip(results, sigmas)):
        #    print("Results for interation {0}: {1} +/- {2}".format(k+1, res, sigma))
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
