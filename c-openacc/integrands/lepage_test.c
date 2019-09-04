#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double lepage_test(double x[], int n) {
    double a = 0.1;
    double pref = pow(1.0/a/sqrt(M_PI), n);
    double coef = 0.0;
    for (int i = 1; i <= 100*n; i++) {
        coef += (float) i;
    }
    for (int i = 0; i < n; i++) {
        coef += pow((x[i] - 1.0/2.0)/a, 2);
    }
    coef -= 100.0*n*(100.0*n+1.0)/2.0;
    return pref*exp(-coef);
}
