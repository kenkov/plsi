SHELL=/bin/bash

main:
	python setup.py build_ext  --inplace

clean:
	rm -r *.so *.c build
