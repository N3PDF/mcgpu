#define M_PI 3.14159265358979323846

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