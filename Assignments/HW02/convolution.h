#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cstddef>

// Computes the result of applying a mask to an image as in the convolution process described in HW02.pdf.
// image is an nxn grid stored in row-major order.
// mask is an mxm grid stored in row-major order.
// Stores the result in output, which is an nxn grid stored in row-major order.
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m);

#endif
