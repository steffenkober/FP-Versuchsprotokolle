all: build/v01.pdf


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

# hier weitere Abhängigkeiten für build/vXX.pdf deklarieren:
build/v01.pdf: build/plot.pdf

build/v01.pdf: FORCE | build
	latexmk $(TeXOptions) v01.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build

FORCE:

.PHONY: all clean