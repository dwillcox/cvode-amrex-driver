PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE

USE_CUDA   ?= FALSE
USE_CUDA_CVODE ?= FALSE

ifeq ($(USE_CUDA), TRUE)
  USE_CUDA_CVODE := TRUE
endif

EBASE = main

EXTERN_SEARCH += .

CVODE_HOME ?= ../CVODE

Bpack   := ./Make.package
Blocs   := .

include Make.CVODE


