#CXX = g++
#CXXFLAGS = -Ofast -fopenmp
CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopt-report=5  -qopt-report-phase=vec -mkl=sequential -qopt-zmm-usage=high -fopenmp


TARGETS = md

all:	$(TARGETS)

clean:
	/bin/rm -f *.o $(TARGETS) *~ core* *.out *.optrpt



