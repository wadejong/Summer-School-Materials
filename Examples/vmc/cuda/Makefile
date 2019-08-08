NVCC = nvcc --ptxas-options="-v"  -O3
CXX = g++
TARGETS = vmc

all:	$(TARGETS)

vmc:	vmc.cu
	$(NVCC) -o $@ $^

.PHONY:
clean:
	/bin/rm -f $(TARGETS) *.o core* *~
