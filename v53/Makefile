all: build/v53.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

# hier Python-Skripte:
build/plot.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/v53.pdf deklarieren:
build/v53.pdf: build/plot.pdf

build/v53.pdf: FORCE | build
	latexmk $(TeXOptions) v53.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build

FORCE:

.PHONY: all clean