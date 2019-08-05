#CXX = g++
#CXXFLAGS = -Ofast -fopenmp

CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2  -mkl=sequential   -qopt-report=5 -qopt-report-phase=vec  -qopt-zmm-usage=high

all:	vmc

clean:
	/bin/rm -f *.o vmc *~ core* *.out *.optrpt


