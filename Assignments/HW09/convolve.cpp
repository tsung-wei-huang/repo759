// Author: Ruochun, Lijing Yang
#include "convolve.h"
#include <cstdio>

using std::size_t;

void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m)
{

  #pragma omp target enter data map(to : image[0 : n * n], output[0 : n * n], mask[0 : m * m])

  #pragma omp target teams distribute parallel for collapse(2)

  for (std::size_t x = 0; x < n; x++)
  {
    for (std::size_t y = 0; y < n; y++)
    {
      output[x * n + y] = 0;
      for (std::size_t i = 0; i < m; i++)
      {
        for (std::size_t j = 0; j < m; j++)
        {
          std::size_t ix = x + i - (m - 1) / 2;
          std::size_t iy = y + j - (m - 1) / 2;
          if ((ix < 0 || ix >= n) && (iy < n && iy >= 0))
          {
            output[x * n + y] += mask[i * m + j];
            continue;
          }
          else if ((ix >= 0 && ix < n) && (iy >= n || iy < 0))
          {
            output[x * n + y] += mask[i * m + j];
            continue;
          }
          else if ((ix < 0 || ix >= n) && (iy >= n || iy < 0))
          {
            continue;
          }

          output[x * n + y] += mask[i * m + j] * image[ix * n + iy];
        }
      }
    }
  }

  #pragma omp target exit data map(from : output[0 : n * n]) map(from : image[0 : n * n], mask[0 : m * m])
}
