#include "definitions.h"
double lepage_integrand(const int n_dim, const double *randoms) {
    const double a = 0.1;
    const double pref = pow(1.0/a/sqrt(M_PI), n_dim);
    double coef = 0.0;
    for (int j = 0; j < n_dim; j++) {
        coef += pow( (randoms[j] - 1.0/2.0)/a, 2 );
    }
    const double lepage = pref*exp(-coef);
    return lepage;
}

//! Represents the state of a particular generator
typedef struct{ uint x; uint c; } mwc64x_state_t;

enum{ MWC64X_A = 4294883355U };
enum{ MWC64X_M = 18446383549859758079UL };

void MWC64X_Step(mwc64x_state_t *s)
{
        uint X=s->x, C=s->c;

        uint Xn=MWC64X_A*X+C;
        uint carry=(uint)(Xn<C);                                // The (Xn<C) will be zero or one for scalar
        uint Cn=mad_hi(MWC64X_A,X,carry);

        s->x=Xn;
        s->c=Cn;
}

// Pre: a<M, b<M
// Post: r=(a+b) mod M
ulong MWC_AddMod64(ulong a, ulong b, ulong M)
{
        ulong v=a+b;
        if( (v>=M) || (v<a) )
                v=v-M;
        return v;
}

// Pre: a<M,b<M
// Post: r=(a*b) mod M
// This could be done more efficently, but it is portable, and should
// be easy to understand. It can be replaced with any of the better
// modular multiplication algorithms (for example if you know you have
// double precision available or something).
ulong MWC_MulMod64(ulong a, ulong b, ulong M)
{
        ulong r=0;
        while(a!=0){
                if(a&1)
                        r=MWC_AddMod64(r,b,M);
                b=MWC_AddMod64(b,b,M);
                a=a>>1;
        }
        return r;
}


// Pre: a<M, e>=0
// Post: r=(a^b) mod M
// This takes at most ~64^2 modular additions, so probably about 2^15 or so instructions on
// most architectures
ulong MWC_PowMod64(ulong a, ulong e, ulong M)
{
        ulong sqr=a, acc=1;
        while(e!=0){
                if(e&1)
                        acc=MWC_MulMod64(acc,sqr,M);
                sqr=MWC_MulMod64(sqr,sqr,M);
                e=e>>1;
        }
        return acc;
}


uint2 MWC_SeedImpl_Mod64(ulong A, ulong M, uint vecSize, uint vecOffset, ulong streamBase, ulong streamGap)
{
        // This is an arbitrary constant for starting LCG jumping from. I didn't
        // want to start from 1, as then you end up with the two or three first values
        // being a bit poor in ones - once you've decided that, one constant is as
        // good as any another. There is no deep mathematical reason for it, I just
        // generated a random number.
        enum{ MWC_BASEID = 4077358422479273989UL };

        ulong dist=streamBase + (get_global_id(0)*vecSize+vecOffset)*streamGap;
        ulong m=MWC_PowMod64(A, dist, M);

        ulong x=MWC_MulMod64(MWC_BASEID, m, M);
        return (uint2)((uint)(x/A), (uint)(x%A));
}

void MWC64X_SeedStreams(mwc64x_state_t *s, ulong baseOffset, ulong perStreamOffset)
{
        uint2 tmp=MWC_SeedImpl_Mod64(MWC64X_A, MWC64X_M, 1, 0, baseOffset, perStreamOffset);
        s->x=tmp.x;
        s->c=tmp.y;
}

//! Return a 32-bit integer in the range [0..2^32)
uint MWC64X_NextUint(mwc64x_state_t *s)
{
        uint res=s->x ^ s->c;
        MWC64X_Step(s);
        return res;
}


double generate_random_array(mwc64x_state_t* rng, const int n_dim, int *seed, __global const double *divisions, double *randoms, int *div_indexes) {
    const double reg_i = 0.0;
    const double reg_f = 1.0;
    double wgt = 1.0;

    for (int j = 0; j < n_dim; j++) {
        const double rn = (double) MWC64X_NextUint(rng)/UINT_MAX;
        const double xn = BINS_MAX*(1.0 - rn);
        int int_xn = max(0, min( (int) xn, BINS_MAX));
        const double aux_rand = xn - int_xn;
        double x_ini = 0.0;
        if (int_xn > 0) {
            x_ini = divisions[BINS_MAX*j + int_xn-1];
        }
        const double xdelta = divisions[BINS_MAX*j + int_xn] - x_ini;
        const double rand_x = x_ini + xdelta*aux_rand;
        wgt *= xdelta*BINS_MAX;
        randoms[j] = reg_i + rand_x*(reg_f - reg_i);
        div_indexes[j] = int_xn;
        }
    return wgt;
}


__kernel void generate_random_array_kernel(const int n_events, const int n_dim, __global const double *divisions, __global double *all_randoms, __global double *all_wgts, __global int *all_div_indexes) {
    const int block_id = get_group_id(0);
    const int thread_id = get_local_id(0);
    const int block_size = get_local_size(0);

    const int index = block_id*block_size + thread_id;
    const int grid_dim = get_num_groups(0);
    const int stride = block_size * grid_dim;

    for (int i = index; i < n_events; i+= stride) {
        const int idx = i*n_dim;
    }
}

// Kernel to be run per event
__kernel void events_kernel(__global const double *divisions, const int n_dim, const int events_per_kernel, const double xjac, __global double *all_res, __global double *all_res2) {
    const int block_id = get_group_id(0);
    const int thread_id = get_local_id(0);
    const int block_size = get_local_size(0);

    const int index = block_id*block_size + thread_id;
    const int grid_dim = get_num_groups(0);
    const int stride = block_size * grid_dim;
    double randoms[MAXDIM];
    int indexes[MAXDIM];
    int seed = index;

    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, 0, index);

    const int idx_res2 = index*BINS_MAX*n_dim;

    all_res[index] = 0.0;
    for (int i = 0; i < events_per_kernel; i++) {
        const double wgt = generate_random_array(&rng, n_dim, &seed, divisions, &randoms, &indexes);
//        printf("rn=%f\n", randoms[0]);
        const double lepage = lepage_integrand(n_dim, &randoms);
        const double tmp = xjac*wgt*lepage;
        all_res[index] += tmp;
        for (int j = 0; j < n_dim; j++) {
            const int idx = idx_res2 + indexes[j]*n_dim + j;
//            printf("idx=%d", indexes[j]);
            all_res2[idx] += pow(tmp,2);
        }
    }
}
