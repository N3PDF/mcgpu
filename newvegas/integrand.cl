// Integrand module
// am integrand is a function with the following signature:
// double integrand(const int n_dim, const double *randoms)
// where n_dim is the number of dimensions and *randoms is the array of random variables (size n_dim)

double integrand(const int n_dim, const double randoms[MAXDIM]) {
    const double a = 0.1;
    const double pref = pow(1.0/a/sqrt(M_PI), n_dim);
    double coef = 0.0;
    for (int j = 0; j < n_dim; j++) {
        coef += pow( (randoms[j] - 1.0/2.0)/a, 2 );
    }
    const double lepage = pref*exp(-coef);
    return lepage;
}
