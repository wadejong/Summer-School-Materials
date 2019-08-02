EIGENPATH = /gpfs/projects/molssi/install/include/
LIBINT2PATH = /gpfs/projects/molssi/install
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

CXX = g++
CXXFLAGS = -Ofast -fopenmp

#CXX = icpc
#CXXFLAGS = -std=c++11 -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopenmp

CPPFLAGS = -I$(EIGENPATH)/include $(LIBINT2INCLUDES)
LDFLAGS = 

scf:	scf.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBINT2PATH)/lib/libint2.a

default:: scf

clean:
	rm -f *.o scf *~ *.out
 
