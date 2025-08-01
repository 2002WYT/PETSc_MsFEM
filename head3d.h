#ifndef HEAD3D_H
#define HEAD3D_H
#include <petscdmda.h>
#include <petscdm.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>
#include <petscksp.h>
#include <mpi.h>

typedef struct user_context {
    PetscInt32 rank;
    PetscInt size;
    PetscInt use_hdf5; // 0: no output, 1: output to HDF5
    PetscInt pg; // Petrov-Galerkin method
    PetscInt piecex, piecey, piecez; // total fine mesh size
    PetscInt Nx, Ny, Nz; // coarse mesh size
    PetscInt nx, ny, nz; // local mesh size
    PetscScalar Hx, Hy, Hz;
    PetscScalar hx, hy, hz;
    PetscLogDouble t1, t2;
    PetscViewer viewer;
    PetscInt example;
    PetscReal epsilon;
    PetscInt nele, nelec, nelel, nnodes, nnodesc, nnodesl;
    DM dmcoarse_node;
    DMDALocalInfo info_coarse;
    Mat pnbd;
    Vec zerobdcoarseDMDA;
    PetscScalar KElocal[8][8];
    PetscReal tsolvelocal, tassemble, tsetzero, tloop, tsetup, tcalculate;
    PetscReal tref, tproject, tsolvecoarse;
} UserCtx;
void print_array(const PetscInt m, const PetscInt n, const PetscScalar *array,
    const PetscInt rank);
void print_array(const PetscInt m, const PetscInt n, const PetscInt *array, 
    const PetscInt rank);
PetscErrorCode Outputbasic(UserCtx *ctx);
PetscErrorCode SetMeshSize(UserCtx *ctx);
PetscScalar uefunc(PetscScalar x, PetscScalar y, PetscScalar z, UserCtx *ctx);
PetscScalar zerofunc(PetscScalar x, PetscScalar y, PetscScalar z, UserCtx *ctx);
PetscScalar kfunc1(PetscScalar x, PetscScalar y, PetscScalar z, UserCtx *ctx);
PetscScalar ffunc1(PetscScalar x, PetscScalar y, PetscScalar z, UserCtx *ctx);
PetscErrorCode setDMDAinit(UserCtx *ctx);
PetscErrorCode setpnbd(UserCtx *ctx);
PetscErrorCode lk1(PetscScalar hx, PetscScalar hy, PetscScalar hz, 
    PetscScalar KE[8][8]);
PetscErrorCode block_points(PetscInt nx, PetscInt ny, PetscInt nz, PetscInt Ii, 
    PetscInt ix, PetscInt iy, PetscInt iz, 
    PetscInt* idx, PetscInt* idxx, PetscInt* idxy, PetscInt* idxz);
PetscErrorCode solvefem3d(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, PetscInt rhs, Mat R, Mat bd);
PetscErrorCode solvefem3dvec(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, Vec &R, Vec bd);
PetscErrorCode get_stiff_3d(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, Mat &A);
PetscErrorCode get_rightterm_3d(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, PetscInt rhs, Mat &B, PetscInt functype);
PetscErrorCode calculate_3d1(UserCtx *ctx, PetscInt nx,
    PetscInt ny, PetscInt nz, PetscInt rhs, Mat &R, Mat bd, Mat &A, Mat &B);
PetscErrorCode get_exact_solution_3d(PetscInt x0, PetscInt xl, 
    PetscInt y0, PetscInt yl, PetscInt z0, PetscInt zl, PetscInt nx, PetscInt ny, 
    PetscInt nz, Vec &Ue, UserCtx *ctx);
PetscErrorCode projectionc2f(PetscInt ix, PetscInt iy, PetscInt iz, UserCtx *ctx,
    PetscInt *fineindex);
PetscErrorCode solvefem3dvecDMDA(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, Vec &R, Vec bd, DM dm);
PetscErrorCode get_exact_solution_3dDMDA(PetscInt x0, PetscInt xl, 
    PetscInt y0, PetscInt yl, PetscInt z0, PetscInt zl, PetscInt nx, PetscInt ny, 
    PetscInt nz, Vec &Ue, UserCtx *ctx, DM dm);

#endif // HEAD3D_H