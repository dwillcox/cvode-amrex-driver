#include <cvode/cvode.h>                  /* prototypes for CVODE fcts., consts.      */
#include <nvector/nvector_cuda.h>         /* access to CUDA N_Vector                */
#include <sunmatrix/sunmatrix_dense.h>    /* access to dense SUNMatrix                */
#include <sunlinsol/sunlinsol_dense.h>    /* access to dense SUNLinearSolver          */
#include <cvode/cvode_direct.h>           /* access to CVDls interface                */
#include <sundials/sundials_types.h>      /* definition of realtype                   */
#include <sundials/sundials_math.h>       /* contains the macros ABS, SUNSQR, and EXP */
#include <AMReX_MultiFab.H>
#include "test_react.H"
#include "test_react_F.H"
#include <iostream>

using namespace amrex;

void do_react(const int* lo, const int* hi,
              amrex::Real* state, const int* s_lo, const int* s_hi,
              const int ncomp, const amrex::Real dt)
{
  const int size_x = hi[0]-lo[0]+1;
  const int size_y = hi[1]-lo[1]+1;
  const int size_z = hi[2]-lo[2]+1;
  const int neqs = 3;

  realtype reltol=1.0e-4, time=0.0e0, tout, state_y[neqs];
  realtype abstol_values[neqs] = {1.e-8, 1.e-14, 1.e-6};
        
  for (int i=lo[0]; i<=hi[0]; i++) {
    for (int j=lo[1]; j<=hi[1]; j++) {
      for (int k=lo[2]; k<=hi[2]; k++) {

        N_Vector y = NULL, yout=NULL;
        N_Vector abstol = NULL;
        SUNMatrix Amat = NULL;
        SUNLinearSolver Linsol = NULL;
        void* cvode_mem = NULL;
        int flag;

        // Create NVectors
        y = N_VNew_Cuda(neqs);
        yout = N_VNew_Cuda(neqs);
        abstol = N_VNew_Cuda(neqs);

        // Initialize y
        for (int n=1; n<=neqs; n++) {
          get_state(state, s_lo, s_hi, &ncomp, &i, &j, &k, &n, &state_y[n-1]);
        }
        set_nvector_cuda(y, state_y, neqs);

        // Initialize abstol
        set_nvector_cuda(abstol, abstol_values, neqs);

        // Initialize CVODE
        cvode_mem = CVodeCreate(CV_BDF);
        flag = CVodeInit(cvode_mem, fun_rhs, time, y);
        flag = CVodeSVtolerances(cvode_mem, reltol, abstol);

        // Initialize Linear Solver
        Amat   = SUNDenseMatrix(neqs, neqs);
        Linsol = SUNDenseLinearSolver(y, Amat);
        flag = CVDlsSetLinearSolver(cvode_mem, Linsol, Amat);
        flag = CVDlsSetJacFn(cvode_mem, fun_jac);
        flag = CVodeSetMaxNumSteps(cvode_mem, 150000);

        // Do Integration
        time = time + static_cast<realtype>(dt);
        flag = CVode(cvode_mem, time, yout, &tout, CV_NORMAL);
        if (flag != CV_SUCCESS) amrex::Abort("Failed integration");

        // Save Final State
        for (int n=1; n<=neqs; n++) {
          set_state(state, s_lo, s_hi, &ncomp, &i, &j, &k, &n, &NV_Ith_S(yout, n-1));
        }

        // Free Memory
        N_VDestroy(y);
        N_VDestroy(yout);
        N_VDestroy(abstol);
        CVodeFree(&cvode_mem);
        SUNMatDestroy(Amat);
        SUNLinSolFree(Linsol);
      }
    }
  }
}


static void set_nvector_cuda(N_Vector vec, realtype* data, sunindextype size)
{
  realtype* vec_host_ptr = N_VGetHostArrayPointer_Cuda(vec);
  for (sunindextype i = 0; i < size; i++) {
    vec_host_ptr[i] = data[i];
  }
  N_VCopyToDevice_Cuda(vec_host_ptr);
}


static int fun_rhs(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{  
  fun_rhs_kernel<<<1,1>>>(t, y, ydot, user_data);
  return 0;
}


__global__ void fun_rhs_kernel(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  NV_Ith_S(ydot, 0) = -.04e0*NV_Ith_S(y, 0) + 1.e4*NV_Ith_S(y, 1)*NV_Ith_S(y, 2);
  NV_Ith_S(ydot, 2) = 3.e7*NV_Ith_S(y, 1)*NV_Ith_S(y, 1);
  NV_Ith_S(ydot, 1) = -NV_Ith_S(ydot, 0)-NV_Ith_S(ydot, 2);
}


static int fun_jac(realtype tn, N_Vector y, N_Vector fy, SUNMatrix J,
                   void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{

  fun_jac_kernel<<<1,1>>>(realtype tn, N_Vector y, N_Vector fy, SUNMatrix J,
                          void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
  return 0;
}


__global__ void fun_jac_kernel(realtype tn, N_Vector y, N_Vector fy, SUNMatrix J,
                               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  const int neqs = 3;
  realtype* Jdata;
  Jdata = SUNDenseMatrix_Data(J);
  
  Jdata[0] = -0.04e0;
  Jdata[1] = 1.e4*NV_Ith_S(y, 2);
  Jdata[2] = 1.e4*NV_Ith_S(y, 1);

  Jdata[6] = 0.0e0;
  Jdata[7] = 6.0e7*NV_Ith_S(y, 1);
  Jdata[8] = 0.0e0;
  
  Jdata[3] = 0.04e0;
  Jdata[4] = -Jdata[1]-Jdata[7];
  Jdata[5] = -Jdata[2];
}
