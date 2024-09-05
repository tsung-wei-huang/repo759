# C++ Memory Management

## Preface

The C++ language specifies the operators `new` and `delete` as the canonical way to allocate memory at runtime. These operators provide some advantages over the `*alloc`  functions used in C, but those methods are still a part of the C++ standard library. This strange dichotomy can be confusing to new C++ programmers. One might wonder... which is better? Is `malloc` still allowed? Can `new` and `malloc` be used interchangeably? What about the standard library; how does it allocate memory?

To make matters more complicated, revisions to the C++ standard have provided users with still more methods of memory management. This article hopes to demistify some of this confusion while providing users with a few guidelines that they can use to create clean, modern, and well-formed C++ programs.

---

## Why use `malloc`?

The most obvious reason that `malloc` and its sibling functions are available in C++ is rather simple: _legacy code_. The inclusion of the C standard library in C++ allows the vast majority of C programs to run unmodified as C++ programs.

---

## Why use `new`?

The `new` operator provides a number of features which are not implicitly provided by `malloc`.

### **Type Safety**

The `*alloc` function family always returns a pointer to raw memory (type `void*`). This pointer must be cast to the target type, and the number of bytes allocated must often be scaled by the `sizeof` operator. Refer to the following example which allocates an array containing 10 `int`s.
```c++
int* arr = (int*)malloc(10 * sizeof(int));
```

The `new` family of operators do away with all of this. For an arbitrary type, `T`, the expression `new T[n]` returns a pointer (type `T*`) to a region of memory large enough to store `n` of that type. Compare the following example to the one above.
```c++
int* arr = new int[10];
```

No casting is required and the size of the region of memory is scaled based on the size of the type automatically.

### **Initialization**

The ability of the `*alloc` family of allocators to initialize memory is extremely limited. `malloc` and `realloc` return uninitialized memory, and `calloc` is only able to initialize a region to the value of an `int`. This isn't suitable for complex types such as _classes_ which need to be constructed before they can be used.

`new` once again provides a remedy for this. For any type `T`, the singleton `new` operator will allocate _and initialize_ the memory containing that object. For example, `new` can create and initialize a `std::string`.
```c++
std::string* s = new std::string("Hello!\n");
```
For any type `T` which can be default-constructed, (meaning `T()` doesn't take any arguments), the array syntax of `new[]` can also be used. The following example creates an array of empty `std::strings`.
 ```c++
std::string* ptr = new std::string[10]();
 ```

### **Overloading**

The C++ standard forbids adding overloads to standard library functions. A program which does not adhere to this rule is ill-formed; this means that overloads of `malloc`, `calloc`, `realloc`, and `free` are completely off limits.

There is no such restriction on `new`, which is an _operator_ in the global namespace. This allows a developer to freely change the allocation behavior for their program. This has a number of uses in optimization that are out of the scope of this article. For the purpose of illustration, consider this (very dangerous) example which modifies `new` to only print out the number of bytes rather than actually allocating the memory.
 ```c++
void* operator ::new(size_t num_bytes) {
	std::cout << "I should have allocated " << num_bytes << " bytes of memory.\n";
	return nullptr;
}
```

---

## Interoperability

### `new` and `malloc` in the same program

In spite of the reasons why one ought to use `new` in C++, there are still cases in which `malloc` might end up in a C++ program.

> ## In terms of compatibility, it is acceptable for a program to contain calls to both `malloc` and `new`.

This is what happens when a C++ program links against a C library. The C library still relies on C-style memory allocation with `malloc`, and the C++ program works just fine using `new`. It is usually considered bad practice, but the two methods of allocation can even coexist inside the same file or function.

### Can pointers allocated with `new` be `free`d?

Absolutely not. 
> ## A pointer which comes from `new` must be cleaned up using `delete`.

Similarly, `delete` does not know how to clean up after `malloc`. There is no guarantee that `malloc` and `new` implement the same allocator. Mix-and-match pointers are undefined behavior and should always be avoided.

---

## Takeaways

Don't use `malloc` (or its friends) when working in C++. For raw memory allocation in C++, `new` and `delete` are the tools of choice. 

---

## Further discussions

- Part 2: Memory management using Modern C++ facilities
- Part 3: Extending C++ memory allocation with additional capabilities (advanced)