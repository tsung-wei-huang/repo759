#ifndef SCAN_H
#define SCAN_H

#include <cstddef>

// Performs an inclusive scan on input array arr and stores
// the result in the output array
// arr and output are arrays of n elements
void scan(const float *arr, float *output, std::size_t n);

#endif
