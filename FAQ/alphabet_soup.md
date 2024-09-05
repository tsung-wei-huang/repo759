
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [A](#A) | [B](#B) | [C](#C) | [D](#D) | [E](#E) | [F](#F) | [G](#G) | [H](#H) | [I](#I) | [J](#J) | [K](#K) | [L](#L) | [M](#M) |
| [N](#N) | [O](#O) | [P](#P) | [Q](#Q) | [R](#R) | [S](#S) | [T](#T) | [U](#U) | [V](#V) | [W](#W) | [X](#X) | [Y](#Y) | [Z](#Z) |

---
## C

#### $ (_cache abbr_.)
The dollar sign character is sometimes used as an abbreviation for ["cache"](#cache). This originates from the similarity in English pronunciation between "cache" and "cash" (that is, money or currency).

#### CC (_CUDA_)
Acronym. _See [**C**ompute **C**apability](#compute-capability)_.


#### Compiler
A tool which converts human-readable source code into a representation closer to [machine code](#machine-code).

#### Compute Capability
A numeric identifier for the set of computational hardware features that a CUDA GPU supports. Newer GPUs have a higher CC, but can typically execute programs compiled for a lower CC.

#### CUDA Core
Another name for a [Streaming Processor](#streaming-processor).


---
## D

#### D$ (_cache_)
Abbreviation for **D**ata [**Cache**](#cache). _See also [$](#-cache-abbr)_.


---
## G

#### GPGPU
Acronym. _See [**G**eneral-**P**urpose **GPU** Computing](general-purpose-gpu-computing)_.

#### General-Purpose GPU Computing
The practice of using GPUs for computing workloads other than graphics. Typically, this is used to accelerate certain [SIMD](#simd)-heavy tasks which were traditionally performed on CPUs.


---
## H

#### HBM
Acronym. _See [**H**igh **B**andwidth **M**emory](#high-bandwidth-memory)_.

#### HBM2
Acronym. _See [**H**igh **B**andwidth **M**emory](#high-bandwidth-memory)_.

#### High Bandwidth Memory
A type of high-speed [DRAM](#dram) which uses 3D-stacked memory dies to achieve higher density and transfer rates than conventional memory. Bus width is as high as 1024 bits per stack (compared to 32 bits per chip in conventional DDR memory). The HBM specification has been extended to include HBM2 and HBM2E with each providing higher bandwidth than the previous generation.

#### High Performance Computing
A computing discipline which relies on the combination of computing resources to achieve performance greater than that of a conventional computer. It may also refer to the practices leveraged to achieve such performance. _Compare_ [_HTC_](#htc).

#### HPC
Acronym. _See [**H**igh **P**erformance **C**omputing](#high-performance-computing)_.

#### HTT
Acronym. _See [**H**yper-**T**hreading **T**echnology](#hyper-threading-technology)_.

#### Hyper-threading Technology
Intel's proprietary implementation of [SMT](#SMT).


---
## I

#### I$ (_cache_)
Abbreviation for **I**nstruction [**Cache**](#cache). _See also [$](#-cache-abbr)_.

#### ILP
Acronym. _See [**I**nstruction-**L**evel **P**arallelism](#instruction-level-parallelism)_.

#### Instruction
The binary representation of a single action which can be completed by a processor. _See also_ [_CISC_](#cisc), [_RISC_](#risc).

#### Instruction-level Parallelism
A form of [parallelism](#parallelism) in which a processor is able to execute more than one [instruction](#instruction) simultaneously. Modern [superscalar](#superscalar) processors do this transparently.

---
## M

#### Machine Code
The binary representation of a computer's [instructions](#instruction) which can be understood by a processor.


---
## S

#### SIMD
Acronym. _See [**S**ingle **I**nstruction, **M**ultiple **D**ata](#single-instruction-multiple-data)_.

#### SIMT
Acronym. _See [**S**ingle **I**nstruction, **M**ultiple **T**hreads](#single-instruction-multiple-threads)_.

#### Simultaneous Multithreading
A CPU feature in which a single [superscalar](#superscalar) processing element executes [instructions](#instruction) from multiple independent threads in order to better saturate the available execution resources.

#### Single Instruction, Multiple Data
An execution model in which a single [instruction](#instruction) operates on multiple lanes of incoming data. _Related to [Vectorization](#vectorization) and [SIMT](#simt)_. _See also [SISD](#sisd) and [MPMD](#mpmd)_.

#### Single Instruction, Multiple Threads
A variant of the [SIMD](#simd) execution model in which multiple threads execute the same instruction in lockstep. Compare with [vectorization](#vectorization) where a single instruction executes on multiple data streams within a single thread. Often used in [GPGPU](#gpgpu) computing.

#### SM
Acronym. _See [**S**treaming **M**ultiprocessor](#streaming-multiprocessor)_.

#### SMT
Acronym. _See [**S**imultaneous **M**ulti**T**hreading](#simultaneous-multithreading)_.

#### SP
Acronym. _See [**S**treaming **P**rocessor](#streaming-processor)_.

#### Streaming Multiprocessor
Part of a CUDA-capable GPU which is made up of some number of [Streaming Processors](#streaming-processor). [Warps](#warp) are scheduled to the SM; the warp's [instruction](#instruction) stream is executed simultaneously on each SP.

#### Streaming Processor
Part of a CUDA-capable GPU. Each SP executes a single thread in CUDA's [SIMT](#simt) model.

#### Supercomputing
The process of using a [supercomputer](#supercomputer) to meet computing goals. _Related to_ [_HPC_](#HPC) _and_ [_HTC_](#HTC).

#### Supercomputer
A collection of individual computing resources which are connected such that they can behave as a single computer.

#### Superscalar
A term describing a processor [architecture](#architecture) which is able to execute more than one [instruction](#instruction) per clock cycle. This is usually implemented by sending different instructions to each execution unit within a single processing element (e.g. Integer [ALU](#alu) + Floating Point ALU).


---
## T

#### Task Parallelism
A type of [parallelism](#parallelism) which involves the simultaneous execution of computational tasks on multiple execution resources. The [instructions](#instruction) performed may be different or the same between tasks, and the data operated on by the task may be different or the same.

#### Thread-level Parallelism
An implementation of [task parallelism](#task-parallelism) which is achieved when multiple hardware threads execute streams of [instructions](#instruction) at the same time.

#### TLP
Acronym. _See [**T**hread-**L**evel **P**arallelism]_.

