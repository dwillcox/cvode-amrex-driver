#include <cvode/cvode.h>                  /* prototypes for CVODE fcts., consts.      */
#include <nvector/nvector_cuda.h>         /* access to CUDA N_Vector                  */
#include <sunlinsol/sunlinsol_spgmr.h>    /* access to SPGMR SUNLinearSolver          */
#include <cvode/cvode_spils.h>            /* access to CVSpils interface              */
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
        flag = CVodeSetMaxNumSteps(cvode_mem, 150000);	

        // Initialize Linear Solver
        Linsol = SUNSPGMR(y, PREC_NONE, 0);
        flag = CVSpilsSetLinearSolver(cvode_mem, Linsol);
	flag = CVSpilsSetJacTimes(cvode_mem, NULL, fun_jac_times_vec);

        // Do Integration
        time = time + static_cast<realtype>(dt);
        flag = CVode(cvode_mem, time, yout, &tout, CV_NORMAL);
        if (flag != CV_SUCCESS) amrex::Abort("Failed integration");

        // Save Final State
        get_nvector_cuda(y, state_y, neqs);	
        for (int n=1; n<=neqs; n++) {
          set_state(state, s_lo, s_hi, &ncomp, &i, &j, &k, &n, &state_y[n-1]);
        }

        // Free Memory
        N_VDestroy(y);
        N_VDestroy(yout);
        N_VDestroy(abstol);
        CVodeFree(&cvode_mem);
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
  N_VCopyToDevice_Cuda(vec);
}


static void get_nvector_cuda(N_Vector vec, realtype* data, sunindextype size)
{
  N_VCopyFromDevice_Cuda(vec);  
  realtype* vec_host_ptr = N_VGetHostArrayPointer_Cuda(vec);
  for (sunindextype i = 0; i < size; i++) {
    data[i] = vec_host_ptr[i];
  }
}


static int fun_rhs(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  realtype* ydot_d = N_VGetDeviceArrayPointer_Cuda(ydot);
  realtype* y_d = N_VGetDeviceArrayPointer_Cuda(y);  
  fun_rhs_kernel<<<1,1>>>(t, y_d, ydot_d, user_data);
  return 0;
}


static int fun_jac_times_vec(N_Vector v, N_Vector Jv, realtype t,
			     N_Vector y, N_Vector fy,
			     void *user_data, N_Vector tmp)
{
  realtype* v_d   = N_VGetDeviceArrayPointer_Cuda(v);
  realtype* Jv_d  = N_VGetDeviceArrayPointer_Cuda(Jv);
  realtype* y_d   = N_VGetDeviceArrayPointer_Cuda(y);
  realtype* fy_d  = N_VGetDeviceArrayPointer_Cuda(fy);
  realtype* tmp_d = N_VGetDeviceArrayPointer_Cuda(tmp);
  fun_jtv_kernel<<<1,1>>>(v_d, Jv_d, t, y_d, fy_d, user_data, tmp_d);
  return 0;
}

__global__ static void fun_rhs_kernel(realtype t, realtype* y, realtype* ydot,
				      void *user_data)
{
  ydot[0] = -.04e0*y[0] + 1.e4*y[1]*y[2];
  ydot[2] = 3.e7*y[1]*y[1];
  ydot[1] = -ydot[0]-ydot[2];
}


__global__ static void fun_jtv_kernel(realtype* v, realtype* Jv, realtype t,
				      realtype* y, realtype* fy,
				      void* user_data, realtype* tmp)
{
  Jv[0] = -0.04e0*v[0] + 1.e4*y[2]*v[1] + 1.e4*y[1]*v[2];
  Jv[2] = 6.0e7*y[1]*v[1];
  Jv[1] = 0.04e0*v[0] + (-1.e4*y[2]-6.0e7*y[1])*v[1] + (-1.e4*y[1])*v[2];
}
