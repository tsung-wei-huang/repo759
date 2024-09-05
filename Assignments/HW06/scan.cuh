// Author: Nic Olsen

#ifndef SCAN_CUH
#define SCAN_CUH

// Performs an *inclusive scan* on the array input and writes the results to the array output.
// The scan should be computed by making calls to your kernel hillis_steele with
// threads_per_block threads per block in a 1D configuration.
// input and output are arrays of length n allocated as managed memory.
//
// Assumptions:
// - n <= threads_per_block * threads_per_block
__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block);

#endif
