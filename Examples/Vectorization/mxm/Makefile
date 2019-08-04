
TARGETS = mxm

CXX = icpc

CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -mkl=sequential -qopt-zmm-usage=high -qopt-report=5 -qopt-report-phase=vec

all:	$(TARGETS)

%.o : %.cc timerstuff.h
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	/bin/rm -f *.o $(TARGETS) core* *~ *.s *.optrpt
