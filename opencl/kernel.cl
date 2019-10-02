#define BINS_MAX 30
#define M_PI 3.14159265358979323846

double lepage_integrand(const int n_dim, __global const double *randoms) {
    double a = 0.1;
    double pref = pow(1.0/a/sqrt(M_PI), n_dim);
    double coef = 0.0;
    for (int j = 0; j < n_dim; j++) {
        coef += pow( (randoms[j] - 1.0/2.0)/a, 2 );
    }
    double lepage = pref*exp(-coef);
    return lepage;
}

// Kernel to be run per event
__kernel void events_kernel(__global const double *all_randoms, __global const double *all_xwgts, int n_dim, int n_events, double xjac, __global double *all_res, __global double *all_res2) {
    int block_id = get_group_id(0);
    int thread_id = get_local_id(0);
    int block_size = get_local_size(0);

    int index = block_id*block_size + thread_id;
    int grid_dim = get_num_groups(0);
    int stride = block_size * grid_dim;

    // shared lepage
    double a = 0.1;
    double pref = pow(1.0/a/sqrt(M_PI), n_dim);
    for (int i = index; i < n_events; i += stride) {
        double wgt = all_xwgts[i]*xjac;
        double lepage = lepage_integrand(n_dim, &all_randoms[i*n_dim]);
        double tmp = wgt*lepage;
        all_res[i] = tmp;
        all_res2[i] = pow(tmp,2);
    }
}

// testing kernel
__kernel void trial() {
    printf("ciao");
}


// Generation of random array
double bad_rand(int* seed) // 1 <= *seed < m
{
    int const a = 16807; //ie 7**5
    int const m = 2147483647; //ie 2**31-1

    *seed = (int) ((long) (*seed * a))%m;
    double rn = (double) *seed / INT_MAX;
    return (rn + 1.0)/2.0;
}

__kernel void generate_random_array_kernel(const int n_events, const int n_dim, __global const double *divisions, __global double *all_randoms, __global double *all_wgts, __global short *all_div_indexes) {
    double reg_i = 0.0;
    double reg_f = 1.0;

    int block_id = get_group_id(0);
    int thread_id = get_local_id(0);
    int block_size = get_local_size(0);

    int index = block_id*block_size + thread_id;
    int grid_dim = get_num_groups(0);
    int stride = block_size * grid_dim;

    int seed = index;
    for (int j = index; j < n_events; j+= stride) {
        double wgt = 1.0;
        for (int i = 0; i < n_dim; i++) {
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
            all_randoms[j*n_dim + i] = reg_i + rand_x*(reg_f - reg_i);
            all_div_indexes[j*n_dim + i] = int_xn;
            }
        all_wgts[j] = wgt;
    }
}
