# Using CVODE

This example program shows how to use CVODE 4.0.0 in an AMReX application.

To build the example, you should have the environment variable
`AMREX_HOME` defined. You should also define the environment variable
`CVODE_HOME` set to the CVODE installation path (containing the
`include` and `lib` directories).

You should also add `$CVODE_HOME/lib` to `LD_LIBRARY_PATH`.

Then just do `make`.

# Building with CUDA

To build the example with CUDA CVODE, use the PGI compiler and do
`make COMP=PGI USE_CUDA=TRUE`.
