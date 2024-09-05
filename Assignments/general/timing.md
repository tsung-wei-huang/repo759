# High-Resolution Timers in C++

You will be asked to time various operations programmed on various HPC hardware. Below are the two methods that we will use for doing this timing. You should use the appropriate method for the programming task (unless the task specifies a timing method explicitly).

### CPU Programs

For timing CPU programs, we will use C++ support for `high_resolution_clock`. Here's an example which benchmarks a trivial calculation. Please note that on a system as powerful as Euler, the calculation used here still won't take long enough to capture reasonable results unless it runs for many iterations. The following method works on any combination of compiler and operating system with C++11 support enabled, so it will be especially helpful for those of you who prefer to develop your code on Windows before moving it to Euler.

```C++
// The std::chrono namespace provides timer functions in C++
#include <chrono>

// std::ratio provides easy conversions between metric units
#include <ratio>

// iostream is not needed for timers, but we need it for cout
#include <iostream>

// not needed for timers, provides std::pow function
#include <cmath>

// Provide some namespace shortcuts
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

// Set some limits for the test
const size_t TEST_SIZE = 1000;
const size_t TEST_MAX = 32;

int main(int argc, char** argv) {
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    double pow3[TEST_MAX];
    duration<double, std::milli> duration_sec;
    
    // Get the starting timestamp
    start = high_resolution_clock::now();
    
    for (size_t i = 0; i < TEST_SIZE; i++) {
        for (size_t j = 0; j < TEST_MAX; j++) {
            pow3[j] = pow(3.f, j);
        }
    }
    
    // Get the ending timestamp
    end = high_resolution_clock::now();
    
    // Print the results for validation
    cout << "Powers of 3:\n";
    for (size_t n = 0; n < TEST_MAX; n++) {
        cout << "3^" << n << " = " << pow3[n] << "\n";
    }
    
    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << "Total time: " << duration_sec.count() << "ms\n";
    
    return 0;
}
```


### GPU Programs
For timing GPU programs, we will use the method outlined in the section _Timing using CUDA Events_ [here](https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/). Essentially, this boils down to the following template:
```C++
cudaEvent_t start;
cudaEvent_t stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

// +---------------+
// | Thing to time |
// +---------------+

cudaEventRecord(stop);
cudaEventSynchronize(stop);

// Get the elapsed time in milliseconds
float ms;
cudaEventElapsedTime(&ms, start, stop);
```
