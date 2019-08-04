CC          = /gpfs/projects/molssi/opt/llvm/bin/clang
OPTFLAGS    = -fopenmp -O3 -fopenmp-targets=nvptx64
LIBS        = -lm

vadd: Solutions/vadd_par.c
	$(CC) $(OPTFLAGS) -o vadd $(LIBS) $<

clean:
	rm -f vadd
