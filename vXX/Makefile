all: build/vXXX.pdf


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

# hier weitere Abhängigkeiten für build/vXXX.pdf deklarieren:
build/vXXX.pdf: build/plot.pdf

build/vXXX.pdf: FORCE | build
	latexmk $(TeXOptions) vXXX.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build

FORCE:

.PHONY: all clean