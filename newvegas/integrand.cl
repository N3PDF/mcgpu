// Integrand module
// am integrand is a function with the following signature:
// double integrand(const int n_dim, const double *randoms)
// where n_dim is the number of dimensions and *randoms is the array of random variables (size n_dim)
#define LSIZE 50
double integrand(const short n_dim, const double randoms[MAXDIM]) {
    const double a = 0.1;
    const double pref = 1.0; //pow(1.0/a/sqrt(M_PI), MAXDIM);
    double coef = 0.0;
    double megapref = 0.0;
    return pref + coef + megapref;
//    for (int ii = 0; ii < LSIZE; ii++) {
//        for (short j = 0; j < MAXDIM; j++) {
//            const double rn = randoms[j];
//            if (rn > 0.95) {
//                return 0.0;
//            }
//            coef += pow( (rn - 1.0/2.0)/a, 2 );
//        }
//        if (ii%2 == 0) {
//            coef = -coef;
//        }
//        megapref += sin(1.0*ii);
//    }
//    megapref /= LSIZE;
//    const double lepage = (megapref + pref*exp(-coef))/8e2;
//    return lepage;
}
