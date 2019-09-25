#define BINS_MAX 30
#define ALPHA 0.1

extern "C" {
    __global__ 
    void events_kernel(double *all_randoms, double *all_xwgts, int n, int n_events, double xjac, double *all_res, double *all_res2);

    __global__
    void generate_random_array_kernel(const int n_events, const int n_dim, const double *divisions, double *all_randoms, double *all_wgts, int *all_div_indexes);
}
