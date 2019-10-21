#define STATE_RNG int
#define RGN_INITIALIZER bad_init
#define GEN_RAN bad_rand

void bad_init(int* seed, int dummy, int index) {
    *seed = index;
}

double bad_rand(int* seed) // 1 <= *seed < m
{
    int const a = 16807; //ie 7**5
    int const m = 2147483647; //ie 2**31-1

    *seed = (int) ((long) (*seed * a))%m;
    double rn = (double) *seed / INT_MAX;
    return (rn + 1.0)/2.0;
}
