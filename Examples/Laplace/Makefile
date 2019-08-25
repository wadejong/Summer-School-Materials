CXX = mpicxx -cxx=g++-9
#CXXFLAGS = -Ofast -fopenmp -fopt-info-vec -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512ifma -mavx512vbmi
CXXFLAGS = -Ofast -fopenmp -fopt-info-vec -mavx2

#CXX = icpc
#CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2  -mkl=sequential   -qopt-report=5 -qopt-report-phase=vec  -qopt-zmm-usage=high

all:	grid

clean:
	/bin/rm -f *.o grid *~ core* *.out *.optrpt err.* log.* 


