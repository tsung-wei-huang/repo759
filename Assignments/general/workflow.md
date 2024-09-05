# Workflow

It's good to develop a comfortable workflow early on in which you work on your assigment on your machine and/or Euler, and then you test before final submission on Euler. When getting a job in the real world, often you face a similar model of having a small personal computer and a powerful shared server system like Euler.

### Sample Workflow
For a starting point, a typical workflow is listed below:
1. Write most of your code using your favorite editor/IDE on your computer (such as [Atom](https://atom.io/) or the cross-platform highly-configurable [Visual Studio Code](https://code.visualstudio.com/)).
1. If your computer has a Nvidia GPU, build and test your code as best you can on your computer until you are convinced that it is doing what it should be doing.
    - If your computer does not, additional resources may be available to you depending on your department. Please refer to technical guides from your department, Piazza discussions, or post a question there. 
    - You could work (edit code) on Euler directly as well.
1. Copy your files over to Euler.
1. Build on Euler.
1. Write a slurm script (using the slurm_usage.md document) that has all of the commands to build and test your code.
1. Submit this job script to slurm to run your script on a compute node.
    - Check the status of your job by `squeue -u <your username>`.
1. Examine the output of your job once it has run (it will be written to a file starting with "slurm-").
1. Use a terminal text editor to make changes to your code if needed, then re-submit your job to test again.
1. Once you are convinced that your code works as expected on Euler, copy your files back to your computer and push/submit them for grading.
    - You can also push/submit your work on Euler.


### Key Skills
Get comfortable as soon as possible with the following:
- connecting to Euler (`ssh`/PuTTY)
- copying files to and from Euler (`scp`/`rsync`/WinSCP, `git`)
- file management on Euler (`cp`, `mv`, `ls`, `cd`, `pwd`, `cat`, etc)
- file editing on Euler with `nano` or `vim`/`emacs` (you can edit remotely with `vs code`)
- customizing a configurable editor, like the ones listed above, to suit your needs
- submitting jobs to build and run your code with slurm on Euler (`sbatch`)
