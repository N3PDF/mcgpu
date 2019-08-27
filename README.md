Main repository for the study of Monte Carlo integration -with importance sampling- with different languages and libraries.

The proposed structure of the repository is as follows:

1. A makefile in the root of the repository that will generate the necessary executables in a bin/ folder

2. For each implementation of the monte carlo a different folder with a name defining what the implementation is, i.e.: fortran-omp with an implementation in fortran

3. Within each folder there must be a folder called integrands with the possible integrands. i.e.: fortran-omp/integrands/pi.f
