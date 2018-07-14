.PHONY: all clean warnings

LATEX = pdflatex
#LATEX = rubber

all: main.pdf

main.pdf:	main.tex someslides.tex

%.pdf: %.tex
	@$(LATEX) -d $<

clean:
	/bin/rm -f *.aux  *.log  *.nav	*.out  *.snm *~  *.toc  

realclean: clean
	/bin/rm -f main.pdf
