These examples compare performance of plain and BLAS implementations of AXPY and GEMM operations.

- If you have an Intel compiler installed, type `make` and execute `./run_axpy.py` and `./run_gemm.py`
  - If you see something like this on a Mac
```
dyld: Library not loaded: @rpath/libmkl_intel_lp64.dylib
  Referenced from: ./gemm
  Reason: image not found
``` 
    you may need to prepend `source /opt/intel/mkl/bin/mklvars.sh intel64; ` to the name of the executable in Python scripts.
- If you do not have an Intel compiler ...

