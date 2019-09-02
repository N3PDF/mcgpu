#define BINS_MAX 50
#define ALPHA 1.5

double internal_rand();
double generate_random_array(const int n_dim, const double *divisions, double *x, int *div_index);
void refine_grid(const double *res_sq, double *subdivisions);
void rebin(const double *rw, const double rc, double *subdivisions);
int vegas(double (*f_integrand)(double*, int), const int warmup, const int n_dim, const int n_iter, const int n_events, double *final_result, double *sigma);
