#include "head3d.h"
#include <petscdmda.h>
#include <petscdm.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>

PetscErrorCode Outputbasic(UserCtx *ctx)
{
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-save", &ctx->use_hdf5, NULL));
    #if defined(PETSC_USE_COMPLEX)
        PetscPrintf(PETSC_COMM_WORLD, "PetscScalar: Complex (PetscComplex)\n");
    #else
        PetscPrintf(PETSC_COMM_WORLD, "PetscScalar: Real (PetscReal)\n");
    #endif
    #if defined(PETSC_USE_64BIT_INDICES)
        PetscPrintf(PETSC_COMM_WORLD, "PetscInt: 64-bit indices\n");
    #else
        PetscPrintf(PETSC_COMM_WORLD, "PetscInt: 32-bit indices\n");
    #endif
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-ex", &ctx->example, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-eps", &ctx->epsilon, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-pg", &ctx->pg, NULL));
    PetscFunctionReturn(0);
}
PetscErrorCode SetMeshSize(UserCtx *ctx)
{
    PetscFunctionBeginUser;

    ctx->piecex = 60, ctx->piecey = 60, ctx->piecez = 60; // total fine mesh size
    ctx->Nx = 10, ctx->Ny = 20, ctx->Nz = 30;     // coarse mesh size
    PetscInt fine = -1, coarse = -1;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-piecex", &ctx->piecex, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-piecey", &ctx->piecey, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-piecez", &ctx->piecez, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nx", &ctx->Nx, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-Ny", &ctx->Ny, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nz", &ctx->Nz, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-fine", &fine, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-coarse", &coarse, NULL));
    if (fine > 0) {
        ctx->piecex = fine;
        ctx->piecey = fine;
        ctx->piecez = fine;
    }
    if (coarse > 0) {
        ctx->Nx = coarse;
        ctx->Ny = coarse;
        ctx->Nz = coarse;
    }
    if (ctx->piecex < ctx->Nx || ctx->piecey < ctx->Ny || ctx->piecez < ctx->Nz) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, 
            "Error: piece size must be larger than coarse mesh size.\n"));
        PetscCall(PetscFinalize());
        return PETSC_ERR_ARG_OUTOFRANGE;
    }
    if (ctx->Nx <= 0 || ctx->Ny <= 0 || ctx->Nz <= 0 || ctx->piecex <= 0 ||
        ctx->piecey <= 0 || ctx->piecez <= 0) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, 
            "Error: mesh size must be positive.\n"));
        PetscCall(PetscFinalize());
        return PETSC_ERR_ARG_OUTOFRANGE;
    }
    if (ctx->piecex % ctx->Nx != 0 || ctx->piecey % ctx->Ny != 0 || 
        ctx->piecez % ctx->Nz != 0) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, 
            "Error: piece size must be divisible by coarse mesh size.\n"));
        PetscCall(PetscFinalize());
        return PETSC_ERR_ARG_OUTOFRANGE;
    }
    PetscPrintf(PETSC_COMM_WORLD, "This is a MsFEM 3D program of elements with "
        "Omega = [0, 1] x [0, 1] x [0, 1].\n"
        "%ld x %ld x %ld fine mesh total and %ld x %ld x %ld coarse mesh.\n", ctx->piecex, 
        ctx->piecey, ctx->piecez, ctx->Nx, ctx->Ny, ctx->Nz);
    PetscPrintf(PETSC_COMM_WORLD, "The example is %ld, and the epsilon is %g.\n" 
        "using Petrov-Galerkin method: %ld\n",
        ctx->example, ctx->epsilon, ctx->pg);
    ctx->nx = ctx->piecex / ctx->Nx;
    ctx->ny = ctx->piecey / ctx->Ny;
    ctx->nz = ctx->piecez / ctx->Nz;
    ctx->nele = ctx->piecex * ctx->piecey * ctx->piecez;
    ctx->nelec = ctx->Nx * ctx->Ny * ctx->Nz;
    ctx->nelel = ctx->nx * ctx->ny * ctx->nz;
    ctx->nnodes = (ctx->piecex + 1) * (ctx->piecey + 1) * (ctx->piecez + 1);
    ctx->nnodesc = (ctx->Nx + 1) * (ctx->Ny + 1) * (ctx->Nz + 1);
    ctx->nnodesl = (ctx->nx + 1) * (ctx->ny + 1) * (ctx->nz + 1);
    ctx->Hx = 1.0 / ctx->Nx;
    ctx->Hy = 1.0 / ctx->Ny;
    ctx->Hz = 1.0 / ctx->Nz;
    ctx->hx = ctx->Hx / ctx->nx;
    ctx->hy = ctx->Hy / ctx->ny;
    ctx->hz = ctx->Hz / ctx->nz;
    PetscFunctionReturn(0);
}
PetscScalar uefunc(PetscScalar x, PetscScalar y, PetscScalar z, UserCtx *ctx)
{
    PetscFunctionBeginUser;
    PetscScalar u = 0.0;
    if (ctx->example == 1 || ctx->example == 2)
    {
        u = x * (1-x) * y * (1-y) * z * (1-z);
    }
    PetscFunctionReturn(u);
}
PetscScalar zerofunc(PetscScalar x, PetscScalar y, PetscScalar z, UserCtx *ctx)
{
    PetscFunctionBeginUser;
    PetscScalar u = 0.0;
    PetscFunctionReturn(u);
}
PetscScalar kfunc1(PetscScalar x, PetscScalar y, PetscScalar z, UserCtx *ctx)
{
    PetscFunctionBeginUser;
    PetscScalar kappa;
    switch (ctx->example)
    {
        case 1:
            kappa = 3.0 + sin(PETSC_PI*x/ctx->epsilon);
            break;
        case 2:
            kappa = 3.0;
            break;
        default:
            kappa = 1.0;
            break;
    }

    PetscFunctionReturn(kappa);
}
PetscScalar ffunc1(PetscScalar x, PetscScalar y, PetscScalar z, UserCtx *ctx)
{
    PetscFunctionBeginUser;
    PetscScalar f = 0.0;
    switch (ctx->example)
    {
        case 1:
            f = (PETSC_PI*cos((PETSC_PI*x)/ctx->epsilon)*(x*y*z*(y - 1)*(z - 1) + 
            y*z*(x - 1)*(y - 1)*(z - 1)))/ctx->epsilon + 
            2*x*y*(sin((PETSC_PI*x)/ctx->epsilon) + 3)*(x - 1)*(y - 1) + 
            2*x*z*(sin((PETSC_PI*x)/ctx->epsilon) + 3)*(x - 1)*(z - 1) + 
            2*y*z*(sin((PETSC_PI*x)/ctx->epsilon) + 3)*(y - 1)*(z - 1);
            break;
        case 2:
            f = 6*x*y*(x - 1)*(y - 1) + 6*x*z*(x - 1)*(z - 1) + 
            6*y*z*(y - 1)*(z - 1);
            break;
        default:
            f = 1.0;
            break;
    }
    PetscFunctionReturn(f);
}
PetscErrorCode setDMDAinit(UserCtx *ctx)
{
    PetscFunctionBeginUser;
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
        DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, ctx->Nx+1, ctx->Ny+1,
        ctx->Nz+1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL,
        NULL, NULL, &ctx->dmcoarse_node));
    PetscCall(DMSetFromOptions(ctx->dmcoarse_node));
    PetscCall(DMSetUp(ctx->dmcoarse_node));
    
    PetscCall(DMDAGetLocalInfo(ctx->dmcoarse_node, &ctx->info_coarse));
    PetscFunctionReturn(0);
}
PetscErrorCode setpnbd(UserCtx *ctx)
{
    PetscFunctionBeginUser;
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 
        (ctx->nx + 1) * (ctx->ny + 1) * (ctx->nz + 1),
        8, 8, NULL, &ctx->pnbd));
    PetscCall(MatSetType(ctx->pnbd, MATSEQDENSE));
    PetscCall(MatSetFromOptions(ctx->pnbd));
    PetscCall(MatSetUp(ctx->pnbd));
    PetscInt Istart, Iend;
    PetscCall(MatGetOwnershipRange(ctx->pnbd, &Istart, &Iend));
    for (PetscInt i = Istart; i < Iend; ++i)
    {
        PetscInt iz = i / ((ctx->nx + 1) * (ctx->ny + 1));
        PetscInt ires = i - iz * (ctx->nx + 1) * (ctx->ny + 1);
        PetscInt ix = ires / (ctx->ny + 1);
        PetscInt iy = ires - ix * (ctx->ny + 1);
        PetscReal xco = (PetscReal)ix / ctx->nx;
        PetscReal yco = (PetscReal)iy / ctx->ny;
        PetscReal zco = (PetscReal)iz / ctx->nz;
        PetscScalar val[8];
        val[0] = (1-xco) * (1-yco) * (1-zco);
        val[1] = (1-xco) *    yco  * (1-zco);
        val[2] =    xco  * (1-yco) * (1-zco);
        val[3] =    xco  *    yco  * (1-zco);
        val[4] = (1-xco) * (1-yco) *    zco;
        val[5] = (1-xco) *    yco  *    zco;
        val[6] =    xco  * (1-yco) *    zco;
        val[7] =    xco  *    yco  *    zco;
        PetscInt col[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        PetscCall(MatSetValues(ctx->pnbd, 1, &i, 8, col, val, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(ctx->pnbd, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(ctx->pnbd, MAT_FINAL_ASSEMBLY));

    PetscCall(DMCreateGlobalVector(ctx->dmcoarse_node, &ctx->zerobdcoarseDMDA));
    PetscCall(VecSet(ctx->zerobdcoarseDMDA, 0.0));
    PetscFunctionReturn(0);
}
PetscErrorCode lk1(PetscScalar hx, PetscScalar hy, PetscScalar hz, 
    PetscScalar KE[8][8])
{
    PetscFunctionBeginUser;
    PetscScalar hx2 = hx * hx, hy2 = hy * hy, hz2 = hz * hz;
    PetscScalar t1 = (hx * hz) / (9 * hy) + (hy * (hx2 / 9 + hz2 / 9)) / (hx * hz);
    PetscScalar t2 = (hy * hz) / (18 * hx) + (hx * (hy2 / 18 - hz2 / 9)) / (hy * hz);
    PetscScalar t3 = (hx * hz) / (18 * hy) + (hy * (hx2 / 18 - hz2 / 9)) / (hx * hz);
    PetscScalar t4 = (hy * (hx2 / 36 - hz2 / 18)) / (hx * hz) - (hx * hz) / (18 * hy);
    PetscScalar t5 = (hx * hz) / (18 * hy) - (hy * (hx2 / 9 - hz2 / 18)) / (hx * hz);
    PetscScalar t6 = (hy * hz) / (36 * hx) - (hx * (hy2 / 18 + hz2 / 18)) / (hy * hz);
    PetscScalar t7 = (hx * hz) / (36 * hy) - (hy * (hx2 / 18 + hz2 / 18)) / (hx * hz);
    PetscScalar t8 = - (hx * hz) / (36 * hy) - (hy * (hx2 / 36 + hz2 / 36)) / (hx * hz);
    PetscScalar KE2[8][8] = {
        {t1, t2, t3, t4, t5, t6, t7, t8},
        {t2, t1, t4, t3, t6, t5, t8, t7},
        {t3, t4, t1, t2, t7, t8, t5, t6},
        {t4, t3, t2, t1, t8, t7, t6, t5},
        {t5, t6, t7, t8, t1, t2, t3, t4},
        {t6, t5, t8, t7, t2, t1, t4, t3},
        {t7, t8, t5, t6, t3, t4, t1, t2},
        {t8, t7, t6, t5, t4, t3, t2, t1}
    };
    for (PetscInt i = 0; i < 8; ++i) {
        for (PetscInt j = 0; j < 8; ++j) {
            KE[i][j] = KE2[i][j];}}
    PetscFunctionReturn(0);
}
PetscErrorCode block_points(PetscInt nx, PetscInt ny, PetscInt nz, PetscInt Ii, 
    PetscInt ix, PetscInt iy, PetscInt iz, 
    PetscInt* idx, PetscInt* idxx, PetscInt* idxy, PetscInt* idxz)
{
    PetscFunctionBeginUser;
    idx[0] = Ii;
    idx[1] = Ii + 1;
    idx[2] = Ii + (ny + 1);
    idx[3] = idx[2] + 1;
    idx[4] = Ii + (nx + 1) * (ny + 1);
    idx[5] = idx[4] + 1;
    idx[6] = idx[4] + (ny + 1);
    idx[7] = idx[6] + 1;

    idxx[0] = ix;
    idxx[1] = ix;
    idxx[2] = ix + 1;
    idxx[3] = ix + 1;
    idxx[4] = ix;
    idxx[5] = ix;
    idxx[6] = ix + 1;
    idxx[7] = ix + 1;

    idxy[0] = iy;
    idxy[1] = iy + 1;
    idxy[2] = iy;
    idxy[3] = iy + 1;
    idxy[4] = iy;
    idxy[5] = iy + 1;
    idxy[6] = iy;
    idxy[7] = iy + 1;
    
    idxz[0] = iz;
    idxz[1] = iz;
    idxz[2] = iz;
    idxz[3] = iz;
    idxz[4] = iz + 1;
    idxz[5] = iz + 1;
    idxz[6] = iz + 1;
    idxz[7] = iz + 1;
    PetscFunctionReturn(0);
}
PetscErrorCode solvefem3d(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, PetscInt rhs, Mat R, Mat bd)
{
    PetscFunctionBeginUser;
    PetscScalar hx = (xl - x0) / nx;
    PetscScalar hy = (yl - y0) / ny;
    PetscScalar hz = (zl - z0) / nz;
    PetscScalar intphi = hx * hy * hz / 8.0;
    Mat A, B;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, (nx + 1) * (ny + 1) * (nz + 1),
        (nx + 1) * (ny + 1) * (nz + 1)));
    PetscCall(MatSetType(A, MATMPIAIJ));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(MatSeqAIJSetPreallocation(A, 27, NULL));
    PetscCall(MatMPIAIJSetPreallocation(A, 27, NULL, 27, NULL));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
    PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, (nx + 1) * (ny + 1) * (nz + 1),
        rhs));
    PetscCall(MatSetType(B, MATDENSE));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatSetUp(B));
    PetscCall(MatZeroEntries(R));
    PetscScalar KE[8][8];
    PetscCall(lk1(hx, hy, hz, KE));
    PetscInt Istart, Iend;
    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    for (PetscInt i = Istart; i < Iend; ++i) 
    {
        PetscInt iz = i / ((nx + 1) * (ny + 1));
        PetscInt ires = i - iz * (nx + 1) * (ny + 1);
        PetscInt ix = ires / (ny + 1);
        PetscInt iy = ires - ix * (ny + 1);
        if (ix == nx || iy == ny || iz == nz) continue;
        PetscInt idx[8], idxx[8], idxy[8], idxz[8];
        PetscCall(block_points(nx, ny, nz, i, ix, iy, iz, idx, idxx, idxy, idxz));
        PetscScalar centercoor[3] = {x0 + (ix+0.5)*hx, 
            y0 + (iy+0.5)*hy, z0 + (iz+0.5)*hz};
        PetscScalar kcenter = 
            kfunc1(centercoor[0], centercoor[1], centercoor[2], ctx);
        PetscScalar fcenter = 
            ffunc1(centercoor[0], centercoor[1], centercoor[2], ctx);
        PetscScalar DDidx[8][8];
        for (PetscInt l = 0; l < 8; l++) {
            if (idxx[l] == 0 || idxx[l] == nx ||
                idxy[l] == 0 || idxy[l] == ny ||
                idxz[l] == 0 || idxz[l] == nz) {
                for (PetscInt ll = 0; ll < 8; ll++) {
                    DDidx[l][ll] = 0.0;
                }
            }
            else {
                for (PetscInt ll = 0; ll < 8; ll++) {
                    DDidx[l][ll] = KE[l][ll] * kcenter;
                }
            }
        } // dirichlet boundary condition
        PetscCall(MatSetValues(A, 8, idx, 8, idx, (const PetscScalar*)DDidx, 
            ADD_VALUES));
        PetscScalar fDD[8];
        for (PetscInt l = 0; l < 8; l++) {
            fDD[l] = fcenter * intphi;
        }
        for (PetscInt l = 0; l < rhs; l++) {
            PetscCall(MatSetValues(B, 8, idx, 1, &l, &fDD[l], ADD_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
        PetscInt iz = Ii / ((nx + 1) * (ny + 1));
        PetscInt ires = Ii - iz * (nx + 1) * (ny + 1);
        PetscInt ix = ires / (ny + 1);
        PetscInt iy = ires - ix * (ny + 1);
        if (ix == 0 || ix == nx || iy == 0 || iy == ny || iz == 0 || iz == nz) {
            PetscScalar oneA = 1.0;
            PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &oneA, INSERT_VALUES));
            for (PetscInt l = 0; l < rhs; l++) {
                PetscScalar valbd = 0.0;
                PetscCall(MatGetValues(bd, 1, &Ii, 1, &l, &valbd));
                PetscCall(MatSetValues(B, 1, &Ii, 1, &l, &valbd, INSERT_VALUES));
            }
        }
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    KSP ksp;
    PC pc;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 1000));
    PetscCall(KSPSetUp(ksp));
    PetscCall(KSPMatSolve(ksp, B, R));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));
    PetscFunctionReturn(0);
}
PetscErrorCode solvefem3dvec(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, Vec &R, Vec bd)
{
    PetscFunctionBeginUser;
    PetscScalar hx = (xl - x0) / nx;
    PetscScalar hy = (yl - y0) / ny;
    PetscScalar hz = (zl - z0) / nz;
    PetscScalar intphi = hx * hy * hz / 8.0;
    Mat A;
    Vec b;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 
        (nx + 1) * (ny + 1) * (nz + 1), 
        (nx + 1) * (ny + 1) * (nz + 1)));
    PetscCall(MatSetType(A, MATMPIAIJ));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE));
    PetscCall(MatSetUp(A));
    PetscCall(MatSeqAIJSetPreallocation(A, 27, NULL));
    PetscCall(MatMPIAIJSetPreallocation(A, 27, NULL, 27, NULL));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
    PetscCall(VecSetSizes(b, 
        PETSC_DECIDE, (nx + 1) * (ny + 1) * (nz + 1)));
    PetscCall(VecSetFromOptions(b));
    PetscCall(VecSetUp(b));
    PetscCall(VecSet(b, 0.0));
    PetscInt sizeb;
    PetscCall(VecGetSize(b, &sizeb));
    PetscScalar KE[8][8];
    PetscCall(lk1(hx, hy, hz, KE));
    PetscInt Istart, Iend, I1, I2;
    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    for (PetscInt i = Istart; i < Iend; ++i) 
    {
        PetscInt iz = i / ((nx + 1) * (ny + 1));
        PetscInt ires = i - iz * (nx + 1) * (ny + 1);
        PetscInt ix = ires / (ny + 1);
        PetscInt iy = ires - ix * (ny + 1);
        if (ix == nx || iy == ny || iz == nz) continue;
        PetscInt idx[8], idxx[8], idxy[8], idxz[8];
        PetscCall(block_points(nx, ny, nz, i, ix, iy, iz, idx, idxx, idxy, idxz));
        PetscScalar centercoor[3] = {x0 + (ix+0.5)*hx, 
            y0 + (iy+0.5)*hy, z0 + (iz+0.5)*hz};
        PetscScalar kcenter = 
            kfunc1(centercoor[0], centercoor[1], centercoor[2], ctx);
        PetscScalar DDidx[8][8];
        for (PetscInt l = 0; l < 8; l++) {
            for (PetscInt ll = 0; ll < 8; ll++) {
                DDidx[l][ll] = KE[l][ll] * kcenter;
            }
        } 
        PetscCall(MatSetValues(A, 8, idx, 8, idx, (const PetscScalar*)DDidx, 
            ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(VecGetOwnershipRange(b, &I1, &I2));
    for (PetscInt i = I1; i < I2; ++i) 
    {
        PetscInt iz = i / ((nx + 1) * (ny + 1));
        PetscInt ires = i - iz * (nx + 1) * (ny + 1);
        PetscInt ix = ires / (ny + 1);
        PetscInt iy = ires - ix * (ny + 1);
        if (ix == nx || iy == ny || iz == nz) continue;
        PetscScalar fcenter = ffunc1(x0 + (ix+0.5)*hx, y0 + (iy+0.5)*hy, 
            z0 + (iz+0.5)*hz, ctx);
        PetscInt idx[8], idxx[8], idxy[8], idxz[8];
        PetscCall(block_points(nx, ny, nz, i, ix, iy, iz, idx, idxx, idxy, idxz));
        PetscScalar fDD[8];
        for (PetscInt l = 0; l < 8; l++) {
            fDD[l] = fcenter * intphi;
        }
        PetscCall(VecSetValues(b, 8, idx, fDD, ADD_VALUES));
    }
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
    PetscInt* zr = new PetscInt[(nx + 1) * (ny + 1) * (nz + 1)];
    PetscInt zcount = 0;
    for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
        PetscInt iz = Ii / ((nx + 1) * (ny + 1));
        PetscInt ires = Ii - iz * (nx + 1) * (ny + 1);
        PetscInt ix = ires / (ny + 1);
        PetscInt iy = ires - ix * (ny + 1);
        if (ix == 0 || ix == nx || iy == 0 || iy == ny || iz == 0 || iz == nz) {
            zr[zcount] = Ii;
            zcount++;
            PetscScalar valbd;
            PetscCall(VecGetValues(bd, 1, &Ii, &valbd));
            PetscCall(VecSetValue(b, Ii, valbd, INSERT_VALUES));
        }
    }
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
    PetscCall(MatZeroRows(A, zcount, zr, 1.0, NULL, NULL));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    KSP ksp;
    PC pc;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 1000));
    PetscCall(KSPSetUp(ksp));
    PetscCall(KSPSolve(ksp, b, R));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    delete [] zr;
    PetscFunctionReturn(0);
}
PetscErrorCode get_stiff_3d(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, Mat &A)
{
    PetscFunctionBeginUser;
    PetscScalar hx = (xl - x0) / nx;
    PetscScalar hy = (yl - y0) / ny;
    PetscScalar hz = (zl - z0) / nz;
    PetscCall(MatZeroEntries(A));
    PetscScalar KE[8][8];
    PetscCall(lk1(hx, hy, hz, KE));
    PetscInt Istart, Iend;
    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    for (PetscInt i = Istart; i < Iend; ++i) 
    {
        PetscInt iz = i / ((nx + 1) * (ny + 1));
        PetscInt ires = i - iz * (nx + 1) * (ny + 1);
        PetscInt ix = ires / (ny + 1);
        PetscInt iy = ires - ix * (ny + 1);
        if (ix == nx || iy == ny || iz == nz) continue;
        PetscInt idx[8], idxx[8], idxy[8], idxz[8];
        PetscCall(block_points(nx, ny, nz, i, ix, iy, iz, idx, idxx, idxy, idxz));
        PetscScalar centercoor[3] = {x0 + (ix+0.5)*hx, 
            y0 + (iy+0.5)*hy, z0 + (iz+0.5)*hz};
        PetscScalar kcenter = 
            kfunc1(centercoor[0], centercoor[1], centercoor[2], ctx);
        PetscScalar DDidx[8][8];
        for (PetscInt l = 0; l < 8; l++) {
            for (PetscInt ll = 0; ll < 8; ll++) {
                DDidx[l][ll] = KE[l][ll] * kcenter;
                
            }
        } // dirichlet boundary condition
        PetscCall(MatSetValues(A, 8, idx, 8, idx, (const PetscScalar*)DDidx, 
            ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(0);
}
PetscErrorCode get_rightterm_3d(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, PetscInt rhs, Mat &B, PetscInt functype)
{
    PetscFunctionBeginUser;
    PetscCall(MatZeroEntries(B));
    PetscScalar hx = (xl - x0) / nx;
    PetscScalar hy = (yl - y0) / ny;
    PetscScalar hz = (zl - z0) / nz;
    PetscScalar intphi = hx * hy * hz / 8.0;
    PetscInt Istart, Iend;
    PetscCall(MatGetOwnershipRange(B, &Istart, &Iend));
    for (PetscInt i = Istart; i < Iend; ++i) 
    {
        PetscInt iz = i / ((nx + 1) * (ny + 1));
        PetscInt ires = i - iz * (nx + 1) * (ny + 1);
        PetscInt ix = ires / (ny + 1);
        PetscInt iy = ires - ix * (ny + 1);
        if (ix == nx || iy == ny || iz == nz) continue;
        PetscInt idx[8], idxx[8], idxy[8], idxz[8];
        PetscCall(block_points(nx, ny, nz, i, ix, iy, iz, idx, idxx, idxy, idxz));
        PetscScalar centercoor[3] = {x0 + (ix+0.5)*hx, 
            y0 + (iy+0.5)*hy, z0 + (iz+0.5)*hz};
        PetscScalar fcenter;
        if (functype == 0) {
            fcenter = 0.0;
        } else if (functype == 1) {
            fcenter = ffunc1(centercoor[0], centercoor[1], centercoor[2], ctx);
        } else {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid functype");
        }
        
        PetscScalar fDD[8];
        for (PetscInt l = 0; l < 8; l++) {
            fDD[l] = fcenter * intphi;
        }
        for (PetscInt l = 0; l < rhs; l++) {
            PetscCall(MatSetValues(B, 8, idx, 1, &l, &fDD[0], ADD_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(0);
}
PetscErrorCode calculate_3d1(UserCtx *ctx, PetscInt nx,
    PetscInt ny, PetscInt nz, PetscInt rhs, Mat &R, Mat bd, Mat &A, Mat &B)
{
    PetscFunctionBeginUser;
    PetscCall(MatZeroEntries(R));
    Mat Acopy;
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Acopy));
    PetscInt Istart, Iend;
    PetscCall(MatGetOwnershipRange(Acopy, &Istart, &Iend));
    PetscInt *bdidx = new PetscInt[(nx + 1) * (ny + 1) * (nz + 1)];
    PetscInt bdcount = 0;
    
    for (PetscInt Ii = Istart; Ii < Iend; Ii++) {
        PetscInt iz = Ii / ((nx + 1) * (ny + 1));
        PetscInt ires = Ii - iz * (nx + 1) * (ny + 1);
        PetscInt ix = ires / (ny + 1);
        PetscInt iy = ires - ix * (ny + 1);
        if (ix == 0 || ix == nx || iy == 0 || iy == ny || iz == 0 || iz == nz) {
            bdidx[bdcount++] = Ii;
            for (PetscInt l = 0; l < rhs; l++) {
                PetscScalar valbd = 0.0;
                PetscCall(MatGetValues(bd, 1, &Ii, 1, &l, &valbd));
                PetscCall(MatSetValues(B, 1, &Ii, 1, &l, &valbd, INSERT_VALUES));
            }
        }
    }
    PetscCall(MatZeroRows(Acopy, bdcount, bdidx, 1.0, NULL, NULL));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    PetscLogDouble t1, t2;
    PetscTime(&t1);
    KSP ksp;
    PC pc;
    PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCall(KSPSetOperators(ksp, Acopy, Acopy));
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCLU));
    // PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST));
    PetscCall(KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 1000));
    PetscCall(KSPSetUp(ksp));
    PetscCall(KSPMatSolve(ksp, B, R));
    PetscTime(&t2);
    ctx->tsolvelocal += t2 - t1;

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&Acopy));
    delete [] bdidx;
    PetscFunctionReturn(0);
}
PetscErrorCode get_exact_solution_3d(PetscInt x0, PetscInt xl, 
    PetscInt y0, PetscInt yl, PetscInt z0, PetscInt zl, PetscInt nx, PetscInt ny, 
    PetscInt nz, Vec &Ue, UserCtx *ctx)
{
    PetscFunctionBeginUser;
    PetscInt I1, I2;
    PetscCall(VecGetOwnershipRange(Ue, &I1, &I2));
    for (PetscInt i = I1; i < I2; ++i) 
    {
        PetscInt iz = i / ((nx + 1) * (ny + 1));
        PetscInt ires = i - iz * (nx + 1) * (ny + 1);
        PetscInt ix = ires / (ny + 1);
        PetscInt iy = ires - ix * (ny + 1);
        if (ix == nx || iy == ny || iz == nz) continue;
        PetscScalar xcoor = x0 + (PetscReal)(ix) / nx * (xl - x0);
        PetscScalar ycoor = y0 + (PetscReal)(iy) / ny * (yl - y0);
        PetscScalar zcoor = z0 + (PetscReal)(iz) / nz * (zl - z0);
        PetscScalar uexact = uefunc(xcoor, ycoor, zcoor, ctx);
        PetscCall(VecSetValue(Ue, i, uexact, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(Ue));
    PetscCall(VecAssemblyEnd(Ue));
    
    PetscFunctionReturn(0);
}
PetscErrorCode get_exact_solution_3dDMDA(PetscInt x0, PetscInt xl, 
    PetscInt y0, PetscInt yl, PetscInt z0, PetscInt zl, PetscInt nx, PetscInt ny, 
    PetscInt nz, Vec &Ue, UserCtx *ctx, DM dm)
{
    PetscFunctionBeginUser;
    DMDALocalInfo info;
    PetscScalar ***uearray;
    PetscCall(DMDAGetLocalInfo(dm, &info));
    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;
    PetscCall(DMDAVecGetArray(dm, Ue, &uearray));
    for (PetscInt iz = zs; iz < zs + zm; ++iz) {
    for (PetscInt iy = ys; iy < ys + ym; ++iy) {
    for (PetscInt ix = xs; ix < xs + xm; ++ix) {
        if (ix == nx || iy == ny || iz == nz) continue;
        PetscScalar xcoor = x0 + (PetscReal)(ix) / nx * (xl - x0);
        PetscScalar ycoor = y0 + (PetscReal)(iy) / ny * (yl - y0);
        PetscScalar zcoor = z0 + (PetscReal)(iz) / nz * (zl - z0);
        PetscScalar uexact = uefunc(xcoor, ycoor, zcoor, ctx);
        uearray[iz][iy][ix] = uexact;
    }}}
    PetscCall(DMDAVecRestoreArray(dm, Ue, &uearray));
    PetscCall(VecAssemblyBegin(Ue));
    PetscCall(VecAssemblyEnd(Ue));
    
    PetscFunctionReturn(0);
}
PetscErrorCode projectionc2f(PetscInt ix, PetscInt iy, PetscInt iz, UserCtx *ctx,
    PetscInt *fineindex)
{
    PetscFunctionBeginUser;
    PetscInt originp = iz * (ctx->nz) * (ctx->piecex + 1) * (ctx->piecey + 1) +
        ix * (ctx->nx) * (ctx->piecey + 1) + iy * (ctx->ny);
    for (PetscInt i = 0; i < (ctx->nx + 1) * (ctx->ny + 1) * (ctx->nz + 1); ++i)
    {
        PetscInt izf = i / ((ctx->nx + 1) * (ctx->ny + 1));
        PetscInt ires = i - izf * (ctx->nx + 1) * (ctx->ny + 1);
        PetscInt ixf = ires / (ctx->ny + 1);
        PetscInt iyf = ires - ixf * (ctx->ny + 1);
        PetscInt gidx = originp + izf * (ctx->piecex + 1) * (ctx->piecey + 1) +
            ixf * (ctx->piecey + 1) + iyf;
        fineindex[i] = gidx;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode solvefem3dvecDMDA(UserCtx *ctx, PetscScalar x0, PetscScalar xl, 
    PetscScalar y0, PetscScalar yl, PetscScalar z0, PetscScalar zl, PetscInt nx,
    PetscInt ny, PetscInt nz, Vec &R, Vec bd, DM dm)
{
    PetscFunctionBeginUser;
    PetscCall(DMCreateGlobalVector(dm, &R));
    PetscScalar hx = (xl - x0) / nx;
    PetscScalar hy = (yl - y0) / ny;
    PetscScalar hz = (zl - z0) / nz;
    PetscScalar intphi = hx * hy * hz / 8.0;
    Mat A;
    Vec b, blocal;
    PetscCall(DMCreateMatrix(dm, &A));
    PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE));
    PetscCall(MatSetUp(A));
    PetscCall(DMCreateGlobalVector(dm, &b));
    PetscCall(VecSet(b, 0.0));
    PetscScalar KE[8][8];
    PetscCall(lk1(hx, hy, hz, KE));
    DMDALocalInfo info;
    PetscCall(DMDAGetLocalInfo(dm, &info));
    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;
    PetscScalar ***barray;
    PetscCall(DMCreateLocalVector(dm, &blocal));
    PetscCall(DMGlobalToLocalBegin(dm, b, ADD_VALUES, blocal));
    PetscCall(DMGlobalToLocalEnd(dm, b, ADD_VALUES, blocal));
    PetscCall(DMDAVecGetArray(dm, blocal, &barray));
    for (PetscInt iz = zs; iz < zs + zm; ++iz) {
    for (PetscInt iy = ys; iy < ys + ym; ++iy) {
    for (PetscInt ix = xs; ix < xs + xm; ++ix) {
        if (ix == nx || iy == ny || iz == nz) continue;
        PetscScalar centercoor[3] = {x0 + (ix+0.5)*hx, 
            y0 + (iy+0.5)*hy, z0 + (iz+0.5)*hz};
        PetscScalar kcenter = 
            kfunc1(centercoor[0], centercoor[1], centercoor[2], ctx);
        PetscScalar fcenter = ffunc1(x0 + (ix+0.5)*hx, y0 + (iy+0.5)*hy, 
            z0 + (iz+0.5)*hz, ctx);
        PetscScalar DDidx[8][8];
        for (PetscInt l = 0; l < 8; l++) {
            for (PetscInt ll = 0; ll < 8; ll++) {
                DDidx[l][ll] = KE[l][ll] * kcenter;
            }
        } 
        PetscScalar fDD[8];
        for (PetscInt l = 0; l < 8; l++) {
            fDD[l] = fcenter * intphi;
        }
        MatStencil rowcol[8];
        rowcol[0].i = ix; rowcol[0].j = iy; rowcol[0].k = iz;
        rowcol[1].i = ix; rowcol[1].j = iy + 1; rowcol[1].k = iz;
        rowcol[2].i = ix + 1; rowcol[2].j = iy; rowcol[2].k = iz;
        rowcol[3].i = ix + 1; rowcol[3].j = iy + 1; rowcol[3].k = iz;
        rowcol[4].i = ix; rowcol[4].j = iy; rowcol[4].k = iz + 1;
        rowcol[5].i = ix; rowcol[5].j = iy + 1; rowcol[5].k = iz + 1;
        rowcol[6].i = ix + 1; rowcol[6].j = iy; rowcol[6].k = iz + 1;
        rowcol[7].i = ix + 1; rowcol[7].j = iy + 1; rowcol[7].k = iz + 1;
        PetscCall(MatSetValuesStencil(A, 8, rowcol, 8, rowcol, 
            (const PetscScalar*)DDidx, ADD_VALUES));
        for (PetscInt l = 0; l < 8; l++) {
            barray[rowcol[l].k][rowcol[l].j][rowcol[l].i] += fDD[l];
        }
    }}}
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(DMDAVecRestoreArray(dm, blocal, &barray));
    PetscCall(DMLocalToGlobalBegin(dm, blocal, ADD_VALUES, b));
    PetscCall(DMLocalToGlobalEnd(dm, blocal, ADD_VALUES, b));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    MatStencil* zr = new MatStencil[(nx + 1) * (ny + 1) * (nz + 1)];
    PetscInt zcount = 0;
    PetscScalar ***Rarray, ***valbdarray;
    PetscCall(DMDAVecGetArray(dm, R, &Rarray));
    PetscCall(DMDAVecGetArrayRead(dm, bd, &valbdarray));
    for (PetscInt iz = zs; iz < zs + zm; ++iz) {
    for (PetscInt iy = ys; iy < ys + ym; ++iy) {
    for (PetscInt ix = xs; ix < xs + xm; ++ix) {
        if (ix == 0 || ix == nx || iy == 0 || iy == ny || iz == 0 || iz == nz) {
            zr[zcount].i = ix; zr[zcount].j = iy; zr[zcount].k = iz;
            zcount++;
            PetscScalar valbd = valbdarray[iz][iy][ix];
            Rarray[iz][iy][ix] = valbd;
        }
    }}}
    PetscCall(DMDAVecRestoreArray(dm, R, &Rarray));
    PetscCall(DMDAVecRestoreArrayRead(dm, bd, &valbdarray));
    PetscCall(VecAssemblyBegin(R));
    PetscCall(VecAssemblyEnd(R));
    PetscCall(MatZeroRowsStencil(A, zcount, zr, 1.0, R, b));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    KSP ksp;
    PC pc;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 1000));
    PetscCall(KSPSetUp(ksp));
    PetscCall(KSPSolve(ksp, b, R));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    delete [] zr;
    PetscFunctionReturn(0);
}