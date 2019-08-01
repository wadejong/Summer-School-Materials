#!/bin/bash
#SBATCH --nodes=2 --tasks-per-node=40 --cpus-per-task=1 --time=00:05:00 --job-name=test -p debug-40core

# Above says:
# - job has 2 (dedicated) nodes with 40 processes per node and one cpu/thread per process with a 5 minute max runtime
# - use the debug-40core queue
# - name the job "test"
# - (slurm by default merges the standard output and error into one file)

# Output should appear in the file "slurm-<jobnumber>.out" in the
# directory from which you submitted the job

# ================================================
# Slurm by default 
# * copies your environment variables from when you submitted the job, so if your
#   modules were correct at time then you don't need to load them here.  If not
#   you should execute the following command
#   source /gpfs/projects/molssi/modules-gnu
# * starts the job running in the same directory that you submitted it from.

# Uncomment this if you want to see other SLURM environment variables
#env | grep SLURM

# Finally, run the executable using #tasks_per_node on each of #nodes
mpirun ./mpihello

# You can run more things below or use different numbers of processes
#mpirun -np 4 ./mpihello
