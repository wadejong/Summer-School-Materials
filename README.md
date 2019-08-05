# Summer-School-Materials

**Wifi connection:** eduroam is available, otherwise connect to SSID WolfieNet-Guest

**Schedule and home page:**  https://sites.google.com/lbl.gov/parcompmolsci/home

Clone this repo using:
~~~
    git clone https://github.com/wadejong/Summer-School-Materials.git
~~~

We will be using the Seawulf cluster at IACS/SBU:

* [Seawulf HPC FAQ etc.](https://it.stonybrook.edu/services/high-performance-computing)

* [Seawulf getting started guide](https://it.stonybrook.edu/help/kb/getting-started-guide)

* [Seawulf SLURM FAQ](https://it.stonybrook.edu/help/kb/using-the-slurm-workload-manager)

* [SLURM reference](https://slurm.schedmd.com/documentation.html)

The login nodes (Intel Haswell processor) of Seawulf can be accessed using ssh
~~~
    ssh <username>@login.seawulf.stonybrook.edu
~~~
but this can be rather busy and is not a very recent CPU.  So, we will also be using `sn-mem` which is a 72-core machine with Intel Skylake processors, 1 NVIDIA P100, and 1.5 TB of memory. This is also accessible from outside the cluster
~~~
    ssh <username>@sn-mem.seawulf.stonybrook.edu
~~~
All other Seawulf nodes are only accessible after you have logged into either the `login` or `sn-mem` nodes.

On Seawulf, software is managed using modules and we will be using several different compilers and tools.  Each time you login please source the appropriate module file (look inside if you want see what is going on) --- pick one of the following:
~~~
    source /gpfs/projects/molssi/modules-gnu
    source /gpfs/projects/molssi/modules-intel
    source /gpfs/projects/molssi/modules-cuda
~~~

