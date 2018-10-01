#ifndef TEST_REACT_H
#define TEST_REACT_H

#include "cvode.h"                  /* prototypes for CVODE fcts., consts.      */
#include "nvector_serial.h"         /* access to serial N_Vector                */
#include "sunmatrix_dense.h"        /* access to dense SUNMatrix                */
#include "sunlinsol_dense.h"        /* access to dense SUNLinearSolver          */
#include "cvode_direct.h"           /* access to CVDls interface                */
#include "sundials_types.h"         /* definition of realtype                   */
#include "sundials_math.h"          /* contains the macros ABS, SUNSQR, and EXP */

void main_main();

void do_react(const int* lo, const int* hi,
              amrex::Real* state, const int* s_lo, const int* s_hi,
              const int ncomp, const amrex::Real dt);

static int fun_rhs(realtype t, N_Vector y, N_Vector ydot, void *user_data);

static int fun_jac(realtype tn, N_Vector y, N_Vector fy, SUNMatrix J,
                   void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

#endif
