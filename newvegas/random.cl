// Interface of any RNG
// Any RNG implemented with the OpenCL MC integrator should expose as preprocessing variables
// - STATE_RNG: a type for storing the state of the random number generator
// - RNG_INITIALIZER: a function with a signature (STATE_T state, ulong base_offset, ulong per_thread_offset)
// - GEN_RAND: a function with a signature (STATE) that returns a double 
#define STATE_RNG mwc64x_state_t
#define RGN_INITIALIZER MWC64X_SeedStreams
#define GEN_RAN MWC64X_NextDouble

// Implementation of the MW64X random number generator


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

double MWC64X_NextDouble(mwc64x_state_t *s)
{
    return (double) MWC64X_NextUint(s)/UINT_MAX;
}
