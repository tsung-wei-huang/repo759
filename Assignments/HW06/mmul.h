// Author: Nic Olsen
#ifndef MMUL_H
#define MMUL_H

#include <cublas_v2.h>

// Uses a single cuBLAS call to perform the operation C := A B + C
// handle is a handle to an open cuBLAS instance
// A, B, and C are matrices with n rows and n columns stored in column-major
// NOTE: The cuBLAS call should be followed by a call to cudaDeviceSynchronize() for timing purposes
void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n);

#endif
