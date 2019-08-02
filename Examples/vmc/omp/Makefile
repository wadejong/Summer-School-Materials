CXX = g++
CXXFLAGS = -Ofast -fopenmp

#CXX = icpc
#CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopenmp

vmc:	vmc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ 

vmc.o:	cmwcran.h

clean:
	/bin/rm -f *.o vmc *~ core* *.out


