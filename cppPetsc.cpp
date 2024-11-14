static char help[] = "Lubrication model for microlayer formation.\n";

/*
   Include "petscdraw.h" so that we can use PETSc drawing routines.
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscsnes.h>
#include <cmath>
#include <boost/math/interpolators/makima.hpp>
#include <iostream>
#include <cstdio>
using boost::math::interpolators::makima;

const double pi = 3.14159265358979323846;

typedef struct
{
    PetscReal rhol = 1684.44, rhov = 0.5850, mu = 6.47e-4,
              sigma = 0.011115, gamma = 7.5e-5, theta = 15 * pi / 180,
              Ls = 1e-8, g = 9.81, Tsat = 298, hfg = 93102, k = 0.0558, Ri = 6.78e-7;

    PetscReal Tw = 300, Ucl = 10.65e-3, Ub = 40e-3;

    PetscReal xcl = 0, l = 0.5e-3, dxReq = 1e-6, dt = 1e-5, t = 0, tend = 0.1;

    PetscInt n = int(l / dxReq) + 1;

    PetscReal dx = l / (n - 1);

    Vec hn, x, xG;

    PetscReal CaBubble = mu * Ub / sigma;

    PetscReal lc = sqrt(sigma / (rhol * g));

    PetscReal hlld = 0.934 * lc * pow(CaBubble, 2. / 3);

    PetscReal rc = lc / sqrt(2);

    PetscReal delta = 0.1 * rc;

} ApplicationCtx;

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
    ApplicationCtx params;

    PetscMPIInt size;
    PetscScalar v, vG, hi;
    PetscReal abstol, rtol, stol;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_SELF, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

    PetscCall(VecCreate(PETSC_COMM_WORLD, &params.x));
    PetscCall(VecSetSizes(params.x, PETSC_DECIDE, params.n));
    PetscCall(VecSetFromOptions(params.x));
    PetscCall(VecDuplicate(params.x, &params.xG));
    PetscCall(VecDuplicate(params.x, &params.hn));

    // Initial Conditions
    PetscReal hlld = params.hlld;
    PetscReal delta = params.delta;
    PetscReal l0 = params.l;

    std::vector<double> xSp = {0, 0.1 * l0, 0.2 * l0, 0.22 * l0, 0.3 * l0, 0.4 * l0, 0.5 * l0, 0.6 * l0, 0.7 * l0, 0.8 * l0, l0};
    std::vector<double> ySp = {0, 1.05 * hlld, 0.52 * hlld, 0.35 * hlld, 0.45 * hlld, 0.59 * hlld, 0.73 * hlld, 0.86 * hlld, 0.95 * hlld, 1.25 * hlld, delta};

    auto spline = makima(std::move(xSp), std::move(ySp));

    for (int j = 0; j < params.n; j++)
    {
        v = (PetscScalar)(params.xcl + j * params.dx);
        hi = (PetscScalar)(spline(params.xcl + j * params.dx));
        VecSetValues(params.x, 1, &j, &v, INSERT_VALUES);
        VecSetValues(params.hn, 1, &j, &hi, INSERT_VALUES);
    }
    PetscCall(VecCopy(params.x, params.xG));
    PetscViewer viewer;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "x.csv", &viewer));
    PetscCall(VecView(params.x, viewer));

    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "h.csv", &viewer));
    PetscCall(VecView(params.hn, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscReal t = params.t;
    PetscReal dt = params.dt;
    PetscReal tend = params.tend;

    PetscReal writeInterval = 1e-2;
    PetscReal lastWritten = t;

    // Time Loop
    while (t < tend)
    {
        t += dt;
        params.t = t;

        params.xcl += params.Ucl * dt;
        params.l += (params.Ub - params.Ucl) * dt;

        std::cout << "t = " << t << std::endl;

        std::cout << "Remeshing" << std::endl;

        // Update Mesh Parameters
        params.n = int(params.l / params.dxReq) + 1;
        params.dx = params.l / (params.n - 1);

        // Remeshing and interpolation

        Vec xOld;
        Vec xNew;
        Vec xNewG;
        Vec hOld;
        Vec hNew;

        PetscCall(VecDuplicate(params.x, &xOld));
        PetscCall(VecDuplicate(params.hn, &hOld));
        PetscCall(VecCopy(params.x, xOld));
        PetscCall(VecCopy(params.hn, hOld));

        PetscCall(VecCreate(PETSC_COMM_WORLD, &xNew));
        PetscCall(VecSetSizes(xNew, PETSC_DECIDE, params.n));
        PetscCall(VecSetFromOptions(xNew));
        PetscCall(VecDuplicate(xNew, &xNewG));
        PetscCall(VecDuplicate(xNew, &hNew));

        const PetscScalar *hOldh, *xOldh;
        PetscScalar *hNewh, *xNewh, *xNewGh;
        PetscCall(VecGetArrayRead(xOld, &xOldh));
        PetscCall(VecGetArray(xNew, &xNewh));
        PetscCall(VecGetArray(xNewG, &xNewGh));
        PetscCall(VecGetArrayRead(hOld, &hOldh));
        PetscCall(VecGetArray(hNew, &hNewh));

        // Update Mesh
        for (int j = 0; j < params.n; j++)
        {
            v = (PetscScalar)(j * params.dx);
            vG = (PetscScalar)(params.xcl + j * params.dx);
            VecSetValues(xNew, 1, &j, &v, INSERT_VALUES);
            VecSetValues(xNewG, 1, &j, &vG, INSERT_VALUES);
        }
        std::cout << "Interpolating onto new mesh." << std::endl;
        // Interpolate onto new Mesh
        for (int j = 0; j < params.n; j++)
        {

            if (j == params.n - 1)
            {
                v = (PetscScalar)(3 * hNewh[params.n - 2] - 3 * hNewh[params.n - 3] + hNewh[params.n - 4]);
                VecSetValues(hNew, 1, &j, &v, INSERT_VALUES);
            }

            else if (j == 0)
            {
                v = (PetscScalar)(0);
                VecSetValues(hNew, 1, &j, &v, INSERT_VALUES);
            }
            else
            {
                if (xNewh[j] > xOldh[j])
                {
                    v = (PetscScalar)(((xOldh[j + 1] - xNewh[j]) * hOldh[j] + (xNewh[j] - xOldh[j]) * hOldh[j + 1]) / (xOldh[j + 1] - xOldh[j]));
                }
                else
                {
                    v = (PetscScalar)(((xOldh[j] - xNewh[j]) * hOldh[j - 1] + (xNewh[j] - xOldh[j - 1]) * hOldh[j]) / (xOldh[j] - xOldh[j - 1]));
                }
                VecSetValues(hNew, 1, &j, &v, INSERT_VALUES);
            }
        }

        PetscCall(VecRestoreArrayRead(hOld, &hOldh));
        PetscCall(VecRestoreArray(hNew, &hNewh));
        PetscCall(VecRestoreArrayRead(xOld, &xOldh));
        PetscCall(VecRestoreArray(xNew, &xNewh));
        PetscCall(VecRestoreArray(xNewG, &xNewGh));

        PetscCall(VecDuplicate(xNew, &params.x));
        PetscCall(VecCopy(xNew, params.x));
        PetscCall(VecDuplicate(xNewG, &params.xG));
        PetscCall(VecCopy(xNewG, params.xG));
        PetscCall(VecDuplicate(hNew, &params.hn));
        PetscCall(VecCopy(hNew, params.hn));

        PetscCall(VecDestroy(&hNew));
        PetscCall(VecDestroy(&xNew));
        PetscCall(VecDestroy(&xNewG));

        PetscCall(VecDestroy(&hOld));
        PetscCall(VecDestroy(&xOld));

        SNES snes; /* SNES context */
        Vec h, r;  /* vectors  h - solution, r - residual*/
        Mat J;     /* Jacobian matrix */
        PetscInt niter, maxit, maxf;

        PetscCall(VecDuplicate(params.hn, &h));
        PetscCall(VecCopy(params.hn, h));
        PetscCall(VecDuplicate(h, &r));

        PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
        PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, params.n, params.n));
        PetscCall(MatSetFromOptions(J));
        PetscCall(MatSeqAIJSetPreallocation(J, 7, NULL));

        PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
        PetscCall(SNESSetFromOptions(snes));
        PetscCall(SNESGetTolerances(snes, &abstol, &rtol, &stol, &maxit, &maxf));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "atol=%g, rtol=%g, stol=%g, maxit=%" PetscInt_FMT ", maxf=%" PetscInt_FMT "\n", (double)abstol, (double)rtol, (double)stol, maxit, maxf));
        PetscCall(SNESSetFunction(snes, r, FormFunction, &params));
        PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, &params));
        std::cout << "Solving nonlinear system of equations." << std::endl;
        // Solve nonlinear system

        PetscCall(SNESSolve(snes, NULL, h));
        PetscCall(SNESGetIterationNumber(snes, &niter));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "number of SNES iterations = %" PetscInt_FMT "\n\n", niter));

        PetscCall(VecCopy(h, params.hn));

        if (t - lastWritten >= writeInterval)
        {
            std::cout << "Writing files." << std::endl;
            lastWritten = t;
            char filename[50]; // Buffer to store the formatted string

            // Use sprintf to format the string with the float
            std::sprintf(filename, "h%.3f.csv", t);

            PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer));
            PetscCall(VecView(params.hn, viewer));
            PetscCall(PetscViewerDestroy(&viewer));

            std::sprintf(filename, "x%.3f.csv", t);
            PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer));
            PetscCall(VecView(params.xG, viewer));
            PetscCall(PetscViewerDestroy(&viewer));
        }

        PetscCall(VecDestroy(&h));
        PetscCall(VecDestroy(&r));
        PetscCall(MatDestroy(&J));
        PetscCall(SNESDestroy(&snes));
    }

    PetscCall(PetscFinalize());
    return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
/* ------------------------------------------------------------------- */
/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ctx - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  f - function vector

   Note:
   The user-defined context can contain any application-specific data
   needed for the function evaluation (such as various parameters, work
   vectors, and grid information).  In this program the context is just
   a vector containing the right-hand side of the discretized PDE.
 */

PetscErrorCode FormFunction(SNES snes, Vec h, Vec f, void *params_)
{
    ApplicationCtx *params = (ApplicationCtx *)params_;
    const PetscScalar *hh, *hhn;
    PetscScalar *ff;
    PetscReal dx = params->dx;
    PetscReal sigma = params->sigma;
    PetscReal mu = params->mu;
    PetscReal Ri = params->Ri;
    PetscReal k = params->k;
    PetscReal rhol = params->rhol;
    PetscReal dt = params->dt;
    PetscReal Tw = params->Tw;
    PetscReal Tsat = params->Tsat;
    PetscReal hfg = params->hfg;
    PetscReal theta = params->theta;
    PetscReal delta = params->delta;
    PetscReal rc = params->rc;
    PetscReal Ls = params->Ls;
    PetscReal Ucl = params->Ucl;
    PetscReal g = params->g;
    PetscInt i, n;
    Vec hn = params->hn;
    PetscFunctionBeginUser;
    /*
       Get pointers to vector data.
         - For default PETSc vectors, VecGetArray() returns a pointer to
           the data array.  Otherwise, the routine is implementation dependent.
         - You MUST call VecRestoreArray() when you no longer need access to
           the array.
    */
    PetscCall(VecGetArrayRead(h, &hh));
    PetscCall(VecGetArrayRead(hn, &hhn));
    PetscCall(VecGetArray(f, &ff));

    /*
       Compute function
    */
    PetscCall(VecGetSize(h, &n));
    ff[0] = hh[0];
    ff[1] = hh[1] - dx * tan(theta);
    ff[2] = (-hh[0] + 3 * hh[1] - 3 * hh[2] + hh[3]);

    ff[n - 1] = hh[n - 1] - delta;
    ff[n - 2] = (hh[n - 1] - 2 * hh[n - 2] + hh[n - 3]) - (dx * dx) / rc;
    ff[n - 3] = (hh[n - 1] - 3 * hh[n - 2] + 3 * hh[n - 3] - hh[n - 4]);

    for (i = 3; i < n - 3; i++)
    {
        ff[i] = hh[i] - hhn[i] + dt / (6 * mu * dx) * (hh[i + 1] * hh[i + 1] * (hh[i + 1] + 3 * Ls) * (sigma * (hh[i + 3] - 2 * hh[i + 2] + 2 * hh[i] - hh[i - 1]) / (2 * dx * dx * dx) + rhol * g) - hh[i - 1] * hh[i - 1] * (hh[i - 1] + 3 * Ls) * (sigma * (hh[i + 1] - 2 * hh[i] + 2 * hh[i - 2] - hh[i - 3]) / (2 * dx * dx * dx) + rhol * g)) - dt * Ucl * (hh[i + 1] - hh[i - 1]) / (2 * dx) + dt * (Tw - Tsat) / (rhol * hfg * (Ri + hh[i] / k)) - dt * sigma * Tsat / (pow(rhol * hfg, 2) * (Ri + hh[i] / k)) * (hh[i + 1] - 2 * hh[i] + hh[i - 1]) / (dx * dx);
    }

    /*
    Restore vectors
    */
    PetscCall(VecRestoreArrayRead(h, &hh));
    PetscCall(VecRestoreArrayRead(hn, &hhn));
    PetscCall(VecRestoreArray(f, &ff));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian(SNES snes, Vec h, Mat jac, Mat B, void *params_)
{
    ApplicationCtx *params = (ApplicationCtx *)params_;
    const PetscScalar *hh;
    PetscReal dx = params->dx;
    PetscReal sigma = params->sigma;
    PetscReal mu = params->mu;
    PetscReal Ri = params->Ri;
    PetscReal k = params->k;
    PetscReal rhol = params->rhol;
    PetscReal dt = params->dt;
    PetscReal Tw = params->Tw;
    PetscReal Tsat = params->Tsat;
    PetscReal hfg = params->hfg;
    PetscReal Ls = params->Ls;
    PetscReal Ucl = params->Ucl;
    PetscReal g = params->g;
    PetscInt i, j[7]; // i : Row Index, j : Column Index
    PetscInt n = params->n;
    PetscScalar A[7];

    PetscFunctionBeginUser;
    /*
       Get pointer to vector data
    */
    PetscCall(VecGetArrayRead(h, &hh));

    /*
       Compute Jacobian entries and insert into matrix.
        - Note that in this case we set all elements for a particular
          row at once.
    */

    /*
       Interior grid points
    */
    for (i = 3; i < n - 3; i++)
    {
        j[0] = i - 3;
        j[1] = i - 2;
        j[2] = i - 1;
        j[3] = i;
        j[4] = i + 1;
        j[5] = i + 2;
        j[6] = i + 3;

        A[0] = dt / (6 * mu * dx) * hh[i - 1] * hh[i - 1] * (hh[i - 1] + 3 * Ls) * sigma / (2 * dx * dx * dx);
        A[1] = -dt / (6 * mu * dx) * hh[i - 1] * hh[i - 1] * (hh[i - 1] + 3 * Ls) * sigma / (dx * dx * dx);
        A[2] = dt / (6 * mu * dx) * (-hh[i + 1] * hh[i + 1] * (hh[i + 1] + 3 * Ls) * sigma / (2 * dx * dx * dx) - 3 * hh[i - 1] * (hh[i - 1] + 2 * Ls) * (sigma * (hh[i + 1] - 2 * hh[i] + 2 * hh[i - 2] - hh[i - 3]) / (2 * dx * dx * dx) + rhol * g)) + Ucl * dt / (2 * dx) - dt * Tsat / (rhol * hfg * rhol * hfg) * (1 / (Ri + hh[i] / k)) / (dx * dx);
        A[3] = 1 + dt / (6 * mu * dx) * (hh[i - 1] * hh[i - 1] * (hh[i - 1] + 3 * Ls) * sigma / (dx * dx * dx) + hh[i + 1] * hh[i + 1] * (hh[i + 1] + 3 * Ls) * sigma / (dx * dx * dx)) - dt * (Tw - Tsat) / (rhol * hfg) * (1 / (Ri + hh[i] / k)) * (1 / (Ri + hh[i] / k)) * 1 / k + dt * Tsat / (rhol * rhol * hfg * hfg) * (1 / (Ri + hh[i] / k)) * (1 / (Ri + hh[i] / k)) * 1 / k * (hh[i + 1] - 2 * hh[i] + hh[i - 1]) / (dx * dx) + dt * Tsat / (rhol * rhol * hfg * hfg) * (1 / (Ri + hh[i] / k)) * 2 / (dx * dx);
        A[4] = dt / (6 * mu * dx) * (-hh[i - 1] * hh[i - 1] * (hh[i - 1] + 3 * Ls) * sigma / (2 * dx * dx * dx) + 3 * hh[i + 1] * (hh[i + 1] + 2 * Ls) * (sigma * (hh[i + 3] - 2 * hh[i + 2] + 2 * hh[i] - hh[i - 1]) / (2 * dx * dx * dx) + rhol * g)) - Ucl * dt / (2 * dx) - dt * Tsat / (rhol * hfg * rhol * hfg) * (1 / (Ri + hh[i] / k)) / (dx * dx);
        A[5] = -dt / (6 * mu * dx) * hh[i + 1] * hh[i + 1] * (hh[i + 1] + 3 * Ls) * sigma / (dx * dx * dx);
        A[6] = dt / (6 * mu * dx) * hh[i + 1] * hh[i + 1] * (hh[i + 1] + 3 * Ls) * sigma / (2 * dx * dx * dx);

        PetscCall(MatSetValues(B, 1, &i, 7, j, A, INSERT_VALUES));
    }

    /*
       Boundary points
    */
    i = 0;
    A[0] = 1.0;

    PetscCall(MatSetValues(B, 1, &i, 1, &i, A, INSERT_VALUES));

    i = 1;
    A[0] = 1.0;

    PetscCall(MatSetValues(B, 1, &i, 1, &i, A, INSERT_VALUES));

    i = 2;
    j[0] = 0;
    j[1] = 1;
    j[2] = 2;
    j[3] = 3;

    A[0] = -1.0;
    A[1] = 3.0;
    A[2] = -3.0;
    A[3] = 1.0;

    PetscCall(MatSetValues(B, 1, &i, 4, j, A, INSERT_VALUES));

    i = n - 1;
    A[0] = 1.0;

    PetscCall(MatSetValues(B, 1, &i, 1, &i, A, INSERT_VALUES));

    i = n - 2;
    j[0] = n - 3;
    j[1] = n - 2;
    j[2] = n - 1;

    A[0] = 1.0;
    A[1] = -2.0;
    A[2] = 1.0;

    PetscCall(MatSetValues(B, 1, &i, 3, j, A, INSERT_VALUES));

    i = n - 3;
    j[0] = n - 4;
    j[1] = n - 3;
    j[2] = n - 2;
    j[3] = n - 1;

    A[0] = -1.0;
    A[1] = 3.0;
    A[2] = -3.0;
    A[3] = 1.0;

    PetscCall(MatSetValues(B, 1, &i, 4, j, A, INSERT_VALUES));

    /*
       Restore vector
    */
    PetscCall(VecRestoreArrayRead(h, &hh));

    /*
       Assemble matrix
    */
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    if (jac != B)
    {
        PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}