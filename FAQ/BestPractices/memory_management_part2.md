# C++ Memory Management

## Modern C++ Techniques

Thanks to the work of WG21 (the International Organization for Standardization's C++ Committee), C++ has evolved since its first official standardization as ISO/IEC 14882:1998 (sometimes called C++98). The term "Modern" in reference to C++ typically refers to the language revisions based on the principle of [RAII](https://en.cppreference.com/w/cpp/language/raii). This includes the standards C++11 and later.

### _Resource Acquisition Is Initialization_ (RAII)

In simplest terms, RAII guarantees that a resource exists for as long as its controlling object exists. In this sense, a "resource" is a feature of a computer which is in limited supply; it could include things such as memory, files, threads, network connections, and so on.

As a practice, RAII allows developers to write code which largely avoids problematic issues such as memory leaks or orphaned resources. For example, complex objects (such as arrays or classes) are allocated when they appear in code and then are cleaned-up once they are no longer visible. The principle of using the scope of an object to control the availability of its underlying resources is called _Scope Bound Resource Management_ (SBRM), and it largely frees the developer from the responsibility to clean up after themselves.

## Vectors

Consider the `std::vector`, an RAII-friendly array-like container class which completely abstracts away the process of dynamically allocating and deallocating its storage pointer.

```c++
void count_to_n(unsigned int n) {

	std::vector<unsigned int> v;
	for (unsigned int i = 0; i < n; i++) {
		v.push_back(i);
	}

	std::cout << v.size();
}
```

In the example above, the function `count_to_n` creates a `std::vector` and fills it with ascending `unsigned int` values. The memory allocated to store these values is first allocated by the vector's constructor when the vector, `v` is declared. Then it is dynamically reallocated as values are added to the vector. When the vector falls out of scope at the end of the function, the vector's destructor automatically frees the vector's storage.

### `std::vector<T>` vs `new T[]`

Thanks to the operator overloading feature of C++, a `std::vector` can behave much like a C-style array. A vector's `operator[]` returns a transparent reference to a location in the vector's storage, so it can be used in exactly the same way as the array indexing operator in C. This allows the vector to work as a stand-in for an array.

Compare the following two examples which populate an array-like object of size `N` with [random values](random_numbers.md#the-c-way):


```c++
std::random_device entropy_source;
std::mt19937 generator(entropy_source()); 
std::uniform_real_distribution<float> dist(-100.0, 100.0);

float* random_values = new float[N];

for (size_t i = 0; i < N; i++) {
	random_values[i] = dist(generator);
}

delete[] random_values;
```

```c++
std::random_device entropy_source;
std::mt19937 generator(entropy_source()); 
std::uniform_real_distribution<float> dist(-100.0, 100.0);

std::vector<float> random_values(N);

for (size_t i = 0; i < random_values.size(); i++) {
	random_values[i] = dist(generator);
}
```

Both examples are very similar, but vector example doesn't require a `delete` and it takes advantage of the vector's built-in ability to return the number of elements it holds with `.size()`

### Pointer Decay

Ubiquitous idiom in high-performance computing is to take advantage of the array ↔ pointer dualism in C (and C++). That is, `*(a + n) == a[n]`. This is called "pointer decay" and it lets a developer treat arrays as pointers when passing arguments to functions.

A `std::vector<T>` does not decay to `T*` in the same way that an array does. Fortunately, `std::vector` provides two capabilities that can be used to fix that issue. First, `std::vector` exposes a method `.data()` which returns a `const` pointer to the vector's underlying storage array. In short, `v.data()[n] == v[n]` so long as the access is read-only. The second capability is a little bit more interesting, and it relies on C++'s reference semantics to get a mutable (non-`const`) pointer to the underlying array.

As mentioned above, a vector's `operator[]` returns a transparent and mutable reference to the target element in its underlying storage array. Reference semantics in C++ make the use of a reference almost identical to that of the original object. Similarly...
> ### A pointer to a C++ reference is identical to a pointer to the original object.
So, something like `&v[n]` is the read-write equivalent of `&(v.data()[n])`. More importantly, `&v[0]` is exactly equal to the decayed pointer of the vector's underlying storage array. 

Given a C-style function:
```c++
// A function which performs a += b on an entire array
void bulk_add(int* a, const int* b, size_t n) {
	for (size_t i = 0; i < n; i++) {
		a[i] += b[i];
	}
}
```

Vectors used for storage could be passed by pointer like this:
```c++
std::vector<int> a_vec;
std::vector<int> b_vec;

... // initialize the vectors somehow, maybe with random numbers

// Call bulk_add on the vectors
bulk_add(&a_vec[0], &b_vec[0], a_vec.size());
```

### C++ Container Comprehensions

One major advantage of `std::vector` over a bare pointer from `new` is the unified API for accessing different types of containers. This allows containers such as vectors, linked lists (`std::list`), and key-value maps (`std::map`) to be accessed and modified using the same syntax.

Compare two `for` loops which print the contents of a container, one which iterates numerically over a vector and another which uses the container's iterator bindings:
```c++
for (int i = 0; i < container.size(); i++) {
	std::cout << container[i] << "\n";
}

for (auto it = container.begin(); it != container.end(); it++) {
	std::cout << *it << "\n";
}
```

The first `for` loop requires a container which implements numerically-ordered access using `operator[]`. The second `for` loop is a bit more wordy, but it works for almost any type of standard C++ container.

In terms of readability, there is an even simpler option for standard containers called the range-based `for` loop:
```c++
for (auto element : container) {
	std::cout << element << "\n";
}
```

The range-based `for` syntax generates machine code that is equivalent to the iterator syntax above, but it is even more clear and concise.