#EIGENPATH = /gpfs/software/eigen-eigen-6b38706d90a9
EIGENPATH = /gpfs/projects/molssi/install/include

CXX = icpc
CXXFLAGS = -std=c++11 -g -Wall
CPPFLAGS = -I$(EIGENPATH)

scf: scf.o
	$(CXX) $^ -o $@

clean:
	rm -f *.o scf
