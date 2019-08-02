EIGENPATH = /gpfs/projects/molssi/install/include
LIBINT2PATH = /gpfs/projects/molssi/install
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

CXX = mpicxx
CXXFLAGS = -Ofast -fopenmp

#CXX = mpiicpc
#CXXFLAGS = -std=c++11 -Ofast -ipo -xHOST -g

CPPFLAGS = -I$(EIGENPATH) $(LIBINT2INCLUDES)
LDFLAGS = $(CXXFLAGS)

scf:	scf.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBINT2PATH)/lib/libint2.a

default:: scf

clean:
	rm -f *.o scf *~ *.out
 
