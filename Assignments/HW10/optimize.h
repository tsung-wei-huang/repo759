#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <cstddef>

// This file provides some macros and utils
// that can be used in your optimizeX functions
// You may change the definition of OP, IDENT, and data_t
// to measure the performance for different optimizations,
// i.e. OP could be + or *; IDENT could be 0 or 1;
// data_t could be int or float.
// You can modify this file for plotting purposes,
// but your code should not depend on *any changes in this file*
// in order to compile. When we grade, this file will still
// be overwritten by a clean copy.

#define OP +
#define IDENT 0

typedef int data_t;

typedef struct vec{
    size_t len;
    data_t *data;
    vec(size_t _l) : len(_l), data(nullptr) {};
} vec;

data_t *get_vec_start(vec *v);

void optimize1(vec *v, data_t *dest);
void optimize2(vec *v, data_t *dest);
void optimize3(vec *v, data_t *dest);
void optimize4(vec *v, data_t *dest);
void optimize5(vec *v, data_t *dest);

#endif
