CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopt-zmm-usage=high -qopt-report=5 -qopt-report-phase=vec

SUBDIRS = bench mc mxm

TARGETS = sum sum_timed

all:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir ; done

clean:
	/bin/rm -f $(TARGETS) a.out *.o *~ *.optrpt *.s
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir clean ; done
