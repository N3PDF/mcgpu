#include <cmath>
#include "definitions.h"

extern "C" {
    void events_kernel(
            const double divisions_in[MAXDIM][BINS_MAX],
            const double randoms_in[MAXDIM][BUFFER_SIZE],
            double results_out[BUFFER_SIZE],
            short indexes_out[MAXDIM][BUFFER_SIZE]
            ) {
// Writing the interface for cpp is a bit more complicated than in opencl but allows for more explicit control
#pragma HLS INTERFACE m_axi port = divisions_in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = randoms_in   offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = results_out  offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = indexes_out  offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = divisions_in bundle = control
#pragma HLS INTERFACE s_axilite port = randoms_in   bundle = control
#pragma HLS INTERFACE s_axilite port = results_out  bundle = control
#pragma HLS INTERFACE s_axilite port = indexes_out  bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        for (int i =0; i < BUFFER_SIZE; i++) {
            results_out[i] = sin(i);
            for (int j = 0; j < MAXDIM; j++) {
                indexes_out[j][i] = i+j;
            }
        }

    }
}
