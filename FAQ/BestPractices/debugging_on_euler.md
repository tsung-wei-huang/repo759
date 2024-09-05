# Interactive Debugging on Euler

Euler's rules make one thing very clear:
> **You should never execute your code on Euler's head node**

... that is, the a user shouldn't run programs that they write on the Login server for Euler.

Normally, a program on Euler is run using a shell script and the `sbatch` command. The `sbatch` interpreter reads its configuration options from special directives within the shell script and executes the script on an automatically selected compute node while writing the results to a file. This is fine for most programs, but sometimes there is simply no way for a program to work without interactively receiving input from its user. In HPC, this comes up most frequently during the development of an application when the user is trying to debug their program.

How, then, would a user be able to use interactive debugging tools such as `gdb`, `lldb`, or `cuda-gdb` on euler? Slurm makes this possible through jobs which run on a compute node and send input/output directly to the user's terminal.

## Restrictions on Interactive Jobs

It should be noted that Euler's limited set of resources makes the use of interactive jobs something of a hot issue. With GPUs, there are more users than there is available hardware. To make matters worse, interactive jobs usually take longer to run than comparable batch jobs (for a variety of reasons which won't be explained here).

As a general rule, Euler's researchers are discouraged from running interactive jobs unless it is absolutely necessary. In classes using Euler, students can outnumber the available GPUs by more than 10 to 1, so they are **forbidden** from running their jobs interactively.

### The Exception

The singular exception to the restriction on interactive jobs is for students who don't have access to their own hardware for debugging. In this case, students may run interactive jobs which meet a certain set of requirements (see below).

## Launching Debugging Sessions

To run a program interactively in Slurm, the `srun` tool (a portmanteau of "Slurm" and "Run") is can be used to execute programs which send their output to and receive input from the user's terminal rather than a script.

Within the scope of classes using Euler, there are three types of `srun` commands allowed:

### `cuda-gdb`
```sh
srun -p instruction -t 30 -s --pty -u -G 1 cuda-gdb [...]
```

### `gdb`
```sh
srun -p instruction -t 30 -s --pty -u gdb [...]
```

### `lldb`
```sh
srun -p instruction -t 30 -s --pty -u lldb [...]
```

Take note of the slurm flags inserted before the debugger command
- **`-p instruction`**: Use the `instruction` partition.
- **`-t 30`**: Run for a maximum of 30 minutes.
- **`-s`**: Allow sharing of resources with other jobs
- **`--pty -u`**: Allocate an unbuffered pseudo-teletype device for the job (makes gdb work properly)
- **`-G 1`**: Allocate exactly 1 GPU for the job. (`cuda-gdb` only)

The commands given above should be used **_verbatim_**, with no modifications before the `[...]` which should be replaced with the arguments to the debugger itself.



