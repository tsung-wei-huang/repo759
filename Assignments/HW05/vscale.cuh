#ifndef VSCALE_CUH
#define VSCALE_CUH

// Scales (element-wise) the elements in array a with the factors stored in array b.
// Writes the scaled results to b.
// Both array a and b have length n.

// Example: a = [-5.0, 2.0, 1.5], b = [0.8, 0.3, 0.6], n = 3
// Resulting array b = [-4.0, 0.6, 0.9]

__global__ void vscale(const float *a, float *b, unsigned int n);

#endif
