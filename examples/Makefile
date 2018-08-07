SUBDIRS = Hartree-Fock md2d vmc

all:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir ; done

clean:
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir clean ; done
