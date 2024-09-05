#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cstddef>
#include <omp.h>

// This function does a parallel version of the convolution process in HW02 task2
// using OpenMP. You may recycle your code from HW02.

// "image" is an n by n grid stored in row-major order.
// "mask" is an m by m grid stored in row-major order.
// "output" stores the result as an n by n grid in row-major order.

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m);

#endif
