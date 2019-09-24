
all: plain_python plain_numba vegas_python vegas_numba vegas_numba_omp vegas_numba_pycuda

plain_python:
	python python/plain_mc_python.py

plain_numba:
	python python/plain_mc_numba.py

vegas_python:
	python python/vegas_mc_python.py

vegas_numba:
	python python/vegas_mc_numba.py

vegas_numba_omp:
	python python/vegas_mc_numba_omp.py

vegas_numba_pycuda:
	python python/vegas_numba_pycuda.py
