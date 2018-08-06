EIGENPATH = /gpfs/projects/molssi/install/include
LIBINT2PATH = /gpfs/projects/molssi/install
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

CXX = icpc
CXXFLAGS = -std=c++11 -xHOST -O3 -no-prec-div -fp-model fast=2 -g
CPPFLAGS = -I$(EIGENPATH)/include $(LIBINT2INCLUDES)
LDFLAGS = $(CXXFLAGS) -L$(LIBINT2PATH)/lib -lint2

default:: scf

scf: scf.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) scf.cc -o scf $(LDFLAGS)

clean:
	rm -f *.o scf
