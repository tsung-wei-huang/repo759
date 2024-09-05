# Random Number Generation in C++

In scientific computing, it often becomes necessary to produce sets of (pseudo-) random numbers. As with many tools which have been updated in C++, the best practice for new programs is to eschew the C-style `rand` function in favor of the new C++ `<random>` library.

For some background on Pseudo-Random Number Generators (PRNGs), read the next section. If you don't care about the background and just want to know how why `rand` isn't recommended, [continue below](#whats-wrong-with-rand). If you just want to see how it works in C++, [jump ahead](#the-c-way).

## Random Number Generators

Computers are largely deterministic in their function. A sane program which is given the same set of inputs will produce the same outputs every time that it is run. This is almost always a _Good Thing_, but it makes the generation of random numbers somewhat difficult. After all, how does a deterministic system produce random results? There are a few ways in which this can be done depending on the needs of the application. 

For cryptography, random numbers need to be "truly" random, at least in the sense that they are unpredictable to an observer. Some CPUs provide sources of _hardware randomness_ which rely on stochastic processes within the CPU in order to generate unpredictable bits. Other systems implement _entropy pools_ which combine physical phenomena in the form of user input, network latency, and other processes to create a pool of random bits.

In scientific domains, however, reproducibility is a vital part of research. Results must be reproducible right down to their randomized components in order to be validated. In this case, applications often rely on Pseudo-random Number Generators, or **PRNG**s. These generators are algorithms which produce random bits based on some complex mathematical operation. This function is "seeded" with some starting value and then called repeatedly to generate new values with each iteration.

## What's wrong with `rand`?

The `rand` function in C (or `std::rand` in C++) has long been plagued by two major problems.

### The typical usage does not generate a uniform distribution

Consider the following idiom:
```c++
const int RANGE = 10;
int random_value = rand() % (RANGE + 1);
```

This idiom returns pseudo-random positive integers between `0` and `RANGE`, but the values aren't uniformly distributed. Depending on the modulus and the number of generated values, this poor distribution can even bias the results of an experiment.

### Some implementations of the PRNG function produce patterns in the lower-order bits

If it is not restricted to a given range, `rand` returns integral values between `0` and the implementation-defined constant `RAND_MAX`. In some implementations, the least significant bits of these integers will begin to show patterns after `RAND_MAX` values have been returned. A particularly heinous pattern is that of the lowest-order bit always being the same every `RAND_MAX` iterations. This means that if the `n`th call to `rand` returns an even number, the `n + RAND_MAX`th call will also return an even number. It becomes even more serious implementations with a very small `RAND_MAX` (Microsoft Windows, for example, defines `RAND_MAX` as `32767`).

These issues can be remedied somewhat by clever manipulation of the output of `rand()`, but there's only so much that can be done with a low-quality algorithm.

## The C++ Way

**C++11** (the revision of the C++ standard published in 2011) introduced a random number generation library which provides adaptors for several PRNG algorithms.

For example, a predefined initialization of the Mersenne Twister algorithm ([Matsumoto and Nishimura, 1998](https://dl.acm.org/doi/10.1145/272991.272995)) can be set up using the steps below:

```c++
int some_seed = 759;
std::mt19937 generator(some_seed);
```

This initializes the engine using a predefined seed, after which new values can be generated using the `()` operator.

```c++
auto pseudorandom_value = generator();
```

### Distributions

A generator can be used directly, but there are some limitations on each generator in terms of the type and range of values that it will produce. It is often more suitable to use one of the predefined adapters for the desired distribution.

```c++
const float minval = -64.0, maxval = 27.0;
std::uniform_real_distribution<float> dist(minval, maxval);

float pseudorandom_float = dist(generator);
```

This method uses the randomness provided by the generator to produce a value within the given distribution. There are many distributions listed in [the documentation](https://en.cppreference.com/w/cpp/header/random) for `<random>`; the generated values can be tailored to a similarly large number of scenarios.

> The `uniform_real_distribution` adapter produces a distribution on an interval of `[minval, maxval)` which may not be the desired behavior. To generate values within a closed interval, the following idiom may be applied to the maximum value of a distribution represented using some floating point type `Real`
> ```c++
> std::uniform_real_distribution<Real>(
> 	minval,
> 	std::nextafter(maxval, std::numeric_limits<Real>::max()
> );
> ```

### Seeding the PRNG

For any given seed of the random number generator, the algorithm will produce a deterministic sequence of values. For this reason, it is often desired to seed the random number generator using a different value on each invocation. To match the idiom commonly used in C, the generator can be seeded using the current time.

```c++
auto seed = std::system_clock::now().time_since_epoch().count();
std::mt19937 generator(seed);
```

For situations which require high-quality randomness, it's possible to use the system's non-deterministic entropy source to produce the seed. 

```c++
std::random_device entropy_source;
std::mt19937 generator(entropy_source());
```

> NOTE: It might seem intuitive to simply use the hardware entropy source to directly generate random values, but system entropy can quickly become exhausted if it is consumed faster than it can be produced. For this reason, it's usually best to use this source only to seed a PRNG.

### A More Complete Example

The following example uses another Mersenne Twister implementation to generate 2¹⁶ random `double`s in the range [10, 20] and then prints the mean of those values.

```c++
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
	const int N = 65536;

	std::random_device entropy_source;

	// 64-bit variant of mt19937 isn't necessary here, but it's just an example
	std::mt19937_64 generator(entropy_source()); 

	std::uniform_real_distribution<double> dist(10., 20.);

	std::vector<double> random_values(N);

	// Write a random value to each slot in N
	for (auto& value : random_values) {
		value = dist(generator);
	}

	// Perform a reduction to get the sum of the values.
	double sum = std::reduce(
		random_values.begin(), 
		random_values.end(),
		0.0
	);

	// Print the mean, should be close to 15.0
	std::cout << sum / static_cast<double>(N) << std::endl;

	return 0;
}
```


