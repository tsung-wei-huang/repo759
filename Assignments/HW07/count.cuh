#ifndef COUNT_CUH
#define COUNT_CUH

#include <thrust/device_vector.h>
// Find the unique integers in the array d_in,
// store these integers in values array in ascending order,
// store the occurrences of these integers in counts array.
// values and counts should have equal length.
// Example:
// d_in = [3, 5, 1, 2, 3, 1]
// Expected output:
// values = [1, 2, 3, 5]
// counts = [2, 1, 2, 1]
void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts);

#endif
