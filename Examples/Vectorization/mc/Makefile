
TARGETS = exptest mc0 mc1 mc2 mc3 mc4 mc5

CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopt-report=5 -qopt-report-phase=vec -mkl=sequential -qopt-zmm-usage=high

#CXX = g++
#CXXFLAGS = -Ofast -march=native -m64 -I$(MKLROOT)/include -fopt-info-vec-all -fopenmp -lmvec -lm

#MKLROOT = /opt/intel/mkl # should be set in environment already
LDFLAGS = -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 

all:	$(TARGETS)

%.o : %.cc timerstuff.h
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	/bin/rm -f *.o $(TARGETS) core* *~ *optrpt
