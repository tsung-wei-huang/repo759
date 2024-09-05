// Author: Jason Zhou

#ifndef REDUCE_CUH
#define REDUCE_CUH

// implements the 'first add during global load' version (Kernel 4) for the
// parallel reduction g_idata is the array to be reduced, and is available on
// the device. g_odata is the array that the reduced results will be written to,
// and is available on the device. expects a 1D configuration. uses only
// dynamically allocated shared memory.
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n);

// the sum of all elements in the *input array should be written to the first
// element of the *input array. calls reduce_kernel repeatedly if needed. _No
// part_ of the sum should be computed on host. *input is an array of length N
// in device memory. *output is an array of length = (number of blocks needed
// for the first call of the reduce_kernel) in device memory. configures the
// kernel calls using threads_per_block threads per block. the function should
// end in a call to cudaDeviceSynchronize for timing purposes
__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block);

#endif
