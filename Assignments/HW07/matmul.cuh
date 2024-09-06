// Author: Nic Olsen, Jason Zhou

#ifndef MATMUL_CUH
#define MATMUL_CUH

// You should implement Tiled Matrix Multiplication discussed in class
// Computes the matrix product C = AB by making 'one' call to 'matmul_kernel'.
// A, B, and C are row-major representations of nxn matrices in managed memory.
// Configures the kernel call using a 2D configuration with blocks of dimensions
// block_dim x block_dim. The function should end in a call to
// cudaDeviceSynchronize for timing purposes.

// Use template to formulate your answer
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim);
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim);
__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim);

#endif