// Integrand module
// am integrand is a function with the following signature:
// double integrand(const int n_dim, const double *randoms)
// where n_dim is the number of dimensions and *randoms is the array of random variables (size n_dim)

#define LSIZE 100000
double integrand(const short n_dim, const double randoms[MAXDIM]) {
    const double a = 0.1;
    const double pref = pow(1.0/a/sqrt(M_PI), MAXDIM);
    double coef = 0.0;
    double megapref = 0.0;
    double factor = 1.0;
    for (int ii = 0; ii < LSIZE; ii++) { 
        for (short j = 0; j < MAXDIM; j++) {
            coef += pow( (randoms[j] - 1.0/2.0)/a, 2 );
        }
        factor *= -1.0;
        coef *= -1.0;
        megapref += factor*ii;
    }
    megapref /= LSIZE;
    const double lepage = (megapref + pref*exp(-coef))/1e3;
    return lepage;
}
