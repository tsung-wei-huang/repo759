# Profiling a GPU program with Nsight Compute

> **Euler NOTE**:
> 
> The older versions of CUDA required to complete this task on Euler are not officially supported. Your mileage may vary.

### What is Nsight Compute?

It's a kernel profiler for CUDA applications that comes with both the interactive and the command line interface. We will use the command line interface (`ncu`) to generate profiling data on Euler, and rely on the interactive interface to visualize the data on your local machine. 


### Profiling a GPU program on Euler

We will use the `vector_addition.cu` code available at the [ME759 repo](https://github.com/DanNegrut/ME759/blob/main/2021Spring/GPU/vector_addition.cu) to demonstrate the profiling process. The following commands will be executed on Euler and **all of them should be included in a Slurm script**.

First, we will use a different CUDA version for the profiling task since CUDA versions after 11 dropped support for the Pascal architecture. You will need to:

```sh
module load nvidia/cuda/10.2.2
```
then compile the code as usual with `nvcc` like:
```sh
nvcc vector_addition.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o vector_addition
```
Note that the `-std=c++17` flag is not supported with CUDA 10, so you'll have to modify your code accordingly if it relies on the more advanced C++ features. Once the executable is ready, we will profile the code (in an `sbatch` script) with:

```sh
ncu -o outFileName --set full ./vector_addition <command line arguments>
```
which will produce a file `outFileName.nsight-cuprof-report` that encodes the profiling results but can only be read with the tool to be introduced in the next section. For more details regarding the usage of `ncu`, refer to the documentation [here](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html). 

### Visualizing the profiling results
After copying back the `outFileName.nsight-cuprof-report` file from Euler, you will need to download and install [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) on your local machine to visualize the profiling results. You don't need to have a GPU on your local machine to use this application, but you may need to create a NVIDIA developer account to initiate the download. If you have installed the CUDA Toolkit before, Nsight Compute should be part of it. 

After the installation, you could simply open the `outFileName.nsight-cuprof-report` file with Nsight Compute, and you should be all set.

The following image demonstrates the profiling results of the `vectorAdd` kernel function we just profiled:
![vectorAdd](vecAdd_prof.png?raw=true)



