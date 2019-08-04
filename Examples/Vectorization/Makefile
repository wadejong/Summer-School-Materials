CXX = icpc
CXXFLAGS = -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopt-zmm-usage=high -qopt-report=5 -qopt-report-phase=vec 

SUBDIRS = bench mc mxm

all:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir ; done

clean:
	/bin/rm -f sum.o a.out *~
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir clean ; done
