
TARGETS = exptest mc0 mc1 mc2 mc3 mc4 mc5

#CXX = icpc
#CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -opt-report -mkl=sequential

CXX = g++

MKLROOT = /opt/intel/mkl
LDFLAGS = -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# try Ofast, O3, O2, O1
CXXFLAGS = -Ofast -march=native -fopt-info-vec-optimized -m64 -I$(MKLROOT)/include

all:	$(TARGETS)

%.o : %.cc timerstuff.h
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	/bin/rm -f *.o $(TARGETS) core* *~ *optrpt
