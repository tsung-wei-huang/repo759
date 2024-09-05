#ifndef REDUCE_H
#define REDUCE_H

#include <cstddef>
#include <omp.h>

// this function should do a parallel reduction with OpenMP to get
// the summation of elements in array "arr" in the range [l, r)
// do as much as you can to improve performance, 
// i.e. use simd directive

float reduce(const float* arr, const size_t l, const size_t r);

#endif
