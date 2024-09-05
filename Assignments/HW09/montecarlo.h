#ifndef MONTECARLO_H
#define MONTECARLO_H

#include <cstddef>
#include <omp.h>

// this function returns the number of points that lay inside
// a circle using OpenMP parallel for. 
// You also need to use the simd directive.

// x - an array of random floats in the range [-radius, radius] with length n.
// y - another array of random floats in the range [-radius, radius] with length n.

int montecarlo(const size_t n, const float *x, const float *y, const float radius);

#endif
