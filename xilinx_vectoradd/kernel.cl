#include "definitions.h"


// Copied form https://www.xilinx.com/html_docs/xilinx2017_4/sdaccel_doc/izc1504034357785.html
__kernel
void vector_add_simple(__global const DOUBLE *a, __global const DOUBLE *b, __global DOUBLE *c) {
    int idx = get_global_id(0);
    c[idx] = a[idx] + b[idx];
}


// Copied from Xilinx repo: getting_started/hello_world/helloworld_ocl/src/vector_addition.clm  
// modified just to take the arguments in the correct order and to take DOUBLE data
#define BUFFER_SIZE 256
#define DATA_SIZE 1024
//TRIPCOUNT indentifier
__constant uint c_len = DATA_SIZE/BUFFER_SIZE;
__constant uint c_size = BUFFER_SIZE;
kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void vector_add_xilinx_repo(global const DOUBLE* a,
                global const DOUBLE* b,
                global  DOUBLE* c,
                       const int n_elements) {
    DOUBLE arrayA[BUFFER_SIZE];
    DOUBLE arrayB[BUFFER_SIZE];
    
    __attribute__((xcl_loop_tripcount(c_len, c_len)))
    for (int i = 0 ; i < n_elements ; i += BUFFER_SIZE) {
        int size = BUFFER_SIZE;
        
        if (i + size > n_elements) size = n_elements - i;

        __attribute__((xcl_loop_tripcount(c_size, c_size)))
        __attribute__((xcl_pipeline_loop(1)))
        readA: for (int j = 0 ; j < size ; j++) {
                arrayA[j] = a[i+j]; }

        __attribute__((xcl_loop_tripcount(c_size, c_size)))
        __attribute__((xcl_pipeline_loop(1)))
        readB: for (int j = 0 ; j < size ; j++) {
                arrayB[j] = b[i+j]; }

        __attribute__((xcl_loop_tripcount(c_size, c_size)))
        __attribute__((xcl_pipeline_loop(1)))
        vadd_writeC: for (int j = 0 ; j < size ; j++) {
                c[i+j] = arrayA[j] + arrayB[j]; }
    }
}
