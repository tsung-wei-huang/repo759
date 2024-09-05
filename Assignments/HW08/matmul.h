#ifndef MATMUL_H
#define MATMUL_H

#include <cstddef>
#include <omp.h>

// This function produces a parallel version of matrix multiplication C = A B using OpenMP. 
// The resulting C matrix should be stored in row-major representation. 
// Use mmul2 from HW02 task3. You may recycle the code from HW02.

// The matrices A, B, and C have dimension n by n and are represented as 1D arrays.

void mmul(const float* A, const float* B, float* C, const std::size_t n);

#endif
