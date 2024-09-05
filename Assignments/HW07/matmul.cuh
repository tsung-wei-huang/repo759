// Author: Deepak and Rakshith

#ifndef MATMUL_CUH
#define MATMUL_CUH

#include <mma.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
void mmul_cuda(const half* A, const half* B, half* C, size_t n, unsigned int threads_per_block);
__global__ void mmul_cuda_kernel(const half* A, const half* B, half* C, size_t n);

void mmul_wmma(const half* A, const half* B, half* C, size_t n, unsigned int threads_per_block);
__global__ void mmul_wmma_kernel(const half* A, const half* B, half* C, size_t n);

void mmul_cublas(const half* A, const half* B, half* C, size_t n);

#endif