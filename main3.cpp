// This version do not have FEM soltions for reference.
// usage: mpirun -n 4 ./main -coarse 32 -fine 256
#include "head3d.h"
// #include <iostream>
using namespace std;

const char *filename = "solution.h5"; // Define the HDF5 filename
int main(int argc, char **argv) {
    UserCtx ctx;
    PetscLogDouble t1, t2, t3, t4, t5;
    Vec uc, uccopy, uec, uef, uf, ufcopy;
    PetscReal uecnorm, uefnorm, diffnormc, diffnormf;
    Mat Ac;
    Vec bc, bclocal, uclocal;
    KSPConvergedReason reason;
    KSP ksp;
    PC pc;
    PetscInt zcount = 0;
    PetscScalar ***bcarray, ***ucarray, ***zerobdcoarsearray;
    PetscInt its;
    
    PetscCall(PetscInitialize(&argc, &argv, (char *)0, (char *)0));
    PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
    PetscTime(&ctx.t1);
    ctx.use_hdf5 = 0;
    ctx.pg = 0;
    ctx.example = 1;
    ctx.epsilon = 1.0/16;
    ctx.tsetup = 0.0;
    ctx.tloop = 0.0;
    ctx.tsolvecoarse = 0.0;
    ctx.tsolvelocal = 0.0;
    ctx.tproject = 0.0;

    PetscTime(&t1);
    PetscCall(Outputbasic(&ctx));
    PetscCall(SetMeshSize(&ctx));
    PetscCall(setDMDAinit(&ctx));
    PetscCall(setpnbd(&ctx));
    PetscCall(lk1(ctx.hx, ctx.hy, ctx.hz, ctx.KElocal));
    // get uec, this is on the coarse mesh DMDA
    PetscCall(DMCreateGlobalVector(ctx.dmcoarse_node, &uec));
    PetscCall(get_exact_solution_3dDMDA(0, 1, 0, 1, 0, 1, 
        ctx.Nx, ctx.Ny, ctx.Nz, uec, &ctx, ctx.dmcoarse_node));
    PetscCall(VecNorm(uec, NORM_2, &uecnorm));
    // get uef, this is on the global mesh but not DMDA order
    PetscCall(VecCreate(PETSC_COMM_WORLD, &uef));
    PetscCall(VecSetSizes(uef, PETSC_DECIDE, 
        (ctx.piecex + 1) * (ctx.piecey + 1) * (ctx.piecez + 1)));
    PetscCall(VecSetType(uef, VECMPI));
    PetscCall(VecSetFromOptions(uef));
    PetscCall(VecSetUp(uef));
    PetscCall(get_exact_solution_3d(0, 1, 0, 1, 0, 1,
        ctx.piecex, ctx.piecey, ctx.piecez, uef, &ctx));
    PetscCall(VecNorm(uef, NORM_2, &uefnorm));
    PetscInt xm = ctx.info_coarse.xm, ym = ctx.info_coarse.ym, zm = ctx.info_coarse.zm;
    PetscInt xs = ctx.info_coarse.xs, ys = ctx.info_coarse.ys, zs = ctx.info_coarse.zs;
    Mat *RR_local = new Mat[(ctx.Nx + 1) * (ctx.Ny + 1) * (ctx.Nz + 1)];
    // 创建每个进程用来储存局部基函数矩阵R的大矩阵RR_global
    PetscCall(DMCreateMatrix(ctx.dmcoarse_node, &Ac));
    PetscCall(MatSetOption(Ac, MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
    PetscCall(MatSetOption(Ac, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE));
    PetscCall(MatSetFromOptions(Ac));
    PetscCall(MatSetUp(Ac));
    PetscCall(MatZeroEntries(Ac));
    PetscCall(DMCreateGlobalVector(ctx.dmcoarse_node, &bc));
    PetscCall(VecSet(bc, 0.0));
    PetscCall(DMCreateLocalVector(ctx.dmcoarse_node, &bclocal));
    PetscCall(DMGlobalToLocalBegin(ctx.dmcoarse_node, bc, ADD_VALUES, bclocal));
    PetscCall(DMGlobalToLocalEnd(ctx.dmcoarse_node, bc, ADD_VALUES, bclocal));
    PetscCall(DMDAVecGetArray(ctx.dmcoarse_node, bclocal, &bcarray));
    PetscTime(&t2);
    ctx.tsetup = t2 - t1;
    MPI_Barrier(PETSC_COMM_WORLD);

    for (PetscInt iz = zs; iz < zs + zm; iz++) {
    for (PetscInt iy = ys; iy < ys + ym; iy++) {
    for (PetscInt ix = xs; ix < xs + xm; ix++) {
        if (ix == ctx.Nx || iy == ctx.Ny || iz == ctx.Nz) continue;
        PetscInt globalindex = iz*ctx.Nx*ctx.Ny + ix*ctx.Ny + iy;
        PetscCall(MatCreate(PETSC_COMM_SELF, &RR_local[globalindex]));
        PetscCall(MatSetSizes(RR_local[globalindex], 
            PETSC_DECIDE, PETSC_DECIDE,
            ctx.nnodesl, 8));
        PetscCall(MatSetType(RR_local[globalindex], MATSEQDENSE));
        PetscCall(MatSetFromOptions(RR_local[globalindex]));
        PetscCall(MatSetUp(RR_local[globalindex]));
        PetscScalar x0 = (PetscScalar)ix / ctx.Nx;
        PetscScalar xl = (PetscScalar)(ix + 1) / ctx.Nx;
        PetscScalar y0 = (PetscScalar)iy / ctx.Ny;
        PetscScalar yl = (PetscScalar)(iy + 1) / ctx.Ny;
        PetscScalar z0 = (PetscScalar)iz / ctx.Nz;
        PetscScalar zl = (PetscScalar)(iz + 1) / ctx.Nz;
        Mat Asmall, tmp, bsmall, m, blocal;
        PetscScalar *DDidx, *fDD;
        PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 
            (ctx.nx + 1) * (ctx.ny + 1) * (ctx.nz + 1), 
            (ctx.nx + 1) * (ctx.ny + 1) * (ctx.nz + 1), 
            27, NULL, &Asmall));
        PetscCall(MatSetOption(Asmall, MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
        PetscCall(MatSetOption(Asmall, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE));
        PetscCall(MatSetFromOptions(Asmall));
        PetscCall(MatSetUp(Asmall));
        PetscCall(get_stiff_3d(&ctx, x0, xl, y0, yl, z0, zl, 
            ctx.nx, ctx.ny, ctx.nz, Asmall));
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 
            (ctx.nx + 1) * (ctx.ny + 1) * (ctx.nz + 1), 8, NULL, &tmp));
        PetscCall(MatSetFromOptions(tmp));
        PetscCall(MatSetUp(tmp));
        PetscCall(get_rightterm_3d(&ctx, x0, xl, y0, yl, z0, zl, 
            ctx.nx, ctx.ny, ctx.nz, 8, tmp, 0));
        PetscCall(calculate_3d1(&ctx, ctx.nx, ctx.ny, ctx.nz, 8, 
            RR_local[globalindex], ctx.pnbd, Asmall, tmp));
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 
            (ctx.nx + 1) * (ctx.ny + 1) * (ctx.nz + 1), 1, NULL, &bsmall));
        PetscCall(MatSetFromOptions(bsmall));
        PetscCall(MatSetUp(bsmall));
        PetscCall(get_rightterm_3d(&ctx, x0, xl, y0, yl, z0, zl, 
            ctx.nx, ctx.ny, ctx.nz, 1, bsmall, 1));
        
        if (ctx.pg == 1) {
            Mat ARtmp;
            PetscCall(MatMatMult(Asmall, RR_local[globalindex], MAT_INITIAL_MATRIX, 
                PETSC_DETERMINE, &ARtmp));
            PetscCall(MatTransposeMatMult(ctx.pnbd, ARtmp, MAT_INITIAL_MATRIX, 
                PETSC_DETERMINE, &m));
            PetscCall(MatTransposeMatMult(ctx.pnbd, bsmall, MAT_INITIAL_MATRIX, 
            PETSC_DETERMINE, &blocal));
            PetscCall(MatDestroy(&ARtmp));
        }
        else {
            PetscCall(MatPtAP(Asmall, RR_local[globalindex], MAT_INITIAL_MATRIX, 
                PETSC_DETERMINE, &m));
            PetscCall(MatTransposeMatMult(RR_local[globalindex], bsmall, 
                MAT_INITIAL_MATRIX, PETSC_DETERMINE, &blocal));
        }
        PetscCall(MatDenseGetArray(m, &DDidx));
        PetscCall(MatDenseGetArray(blocal, &fDD));
        MatStencil rowcol[8];
        rowcol[0].i = ix; rowcol[0].j = iy; rowcol[0].k = iz;
        rowcol[1].i = ix; rowcol[1].j = iy + 1; rowcol[1].k = iz;
        rowcol[2].i = ix + 1; rowcol[2].j = iy; rowcol[2].k = iz;
        rowcol[3].i = ix + 1; rowcol[3].j = iy + 1; rowcol[3].k = iz;
        rowcol[4].i = ix; rowcol[4].j = iy; rowcol[4].k = iz + 1;
        rowcol[5].i = ix; rowcol[5].j = iy + 1; rowcol[5].k = iz + 1;
        rowcol[6].i = ix + 1; rowcol[6].j = iy; rowcol[6].k = iz + 1;
        rowcol[7].i = ix + 1; rowcol[7].j = iy + 1; rowcol[7].k = iz + 1;
        PetscCall(MatSetValuesStencil(Ac, 8, rowcol, 
            8, rowcol, (const PetscScalar*)DDidx, ADD_VALUES));
        for (PetscInt l = 0; l < 8; ++l) {
            bcarray[rowcol[l].k][rowcol[l].j][rowcol[l].i] += fDD[l];
        }
        PetscCall(MatDestroy(&tmp));
        PetscCall(MatDestroy(&bsmall));
        PetscCall(MatDestroy(&Asmall));
        PetscCall(MatDestroy(&m));
        PetscCall(MatDestroy(&blocal));
    }}}
    MPI_Barrier(PETSC_COMM_WORLD);
    PetscCall(MatAssemblyBegin(Ac, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Ac, MAT_FINAL_ASSEMBLY));
    PetscCall(DMDAVecRestoreArray(ctx.dmcoarse_node, bclocal, &bcarray));
    PetscCall(DMLocalToGlobalBegin(ctx.dmcoarse_node, bclocal, ADD_VALUES, bc));
    PetscCall(DMLocalToGlobalEnd(ctx.dmcoarse_node, bclocal, ADD_VALUES, bc));
    PetscCall(DMCreateGlobalVector(ctx.dmcoarse_node, &uc));
    PetscCall(VecSet(uc, 0.0));
    PetscCall(VecDuplicate(uc, &uccopy));
    PetscCall(DMDAVecGetArray(ctx.dmcoarse_node, uc, &ucarray));
    PetscCall(DMDAVecGetArrayRead(ctx.dmcoarse_node, ctx.zerobdcoarseDMDA, 
        &zerobdcoarsearray));
    MatStencil *zerorows = new MatStencil[ctx.nnodesc];
    for (PetscInt iz = zs; iz < zs + zm; iz++) {
    for (PetscInt iy = ys; iy < ys + ym; iy++) {
    for (PetscInt ix = xs; ix < xs + xm; ix++) {
        if (ix == 0 || ix == ctx.Nx || 
            iy == 0 || iy == ctx.Ny || 
            iz == 0 || iz == ctx.Nz) {
            zerorows[zcount].i = ix; 
            zerorows[zcount].j = iy;
            zerorows[zcount].k = iz;
            zcount++;
            ucarray[iz][iy][ix] = zerobdcoarsearray[iz][iy][ix];
        }
    }}}
    PetscCall(DMDAVecRestoreArray(ctx.dmcoarse_node, uc, &ucarray));
    PetscCall(DMDAVecRestoreArrayRead(ctx.dmcoarse_node, ctx.zerobdcoarseDMDA, 
        &zerobdcoarsearray));
    PetscCall(VecAssemblyBegin(uc));
    PetscCall(VecAssemblyEnd(uc));
    MatZeroRowsStencil(Ac, zcount, zerorows, 1.0, uc, bc);
    PetscTime(&t3);
    ctx.tloop = t3 - t2;
    /*PetscReal norma, normb;
    PetscCall(MatNorm(Ac, NORM_FROBENIUS, &norma));
    PetscCall(VecNorm(bc, NORM_2, &normb));
    PetscPrintf(PETSC_COMM_WORLD, 
        "Norm of Ac: %g, Norm of bc: %g\n", (double)norma, (double)normb);*/
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, Ac, Ac));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    PetscCall(KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 1000));
    PetscCall(KSPSetUp(ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMG));
    PetscCall(KSPSolve(ksp, bc, uc));
    KSPGetIterationNumber(ksp, &its);
    PetscPrintf(PETSC_COMM_WORLD, "Iterations to convergence: %ld\n", its);
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscTime(&t4);
    ctx.tsolvecoarse = t4 - t3;
    PetscCall(VecCopy(uc, uccopy));
    PetscCall(VecAXPY(uccopy, -1.0, uec)); // uccopy = uccopy - uec
    PetscCall(VecNorm(uccopy, NORM_2, &diffnormc));

    // now we solve the fine mesh problem
    // uf is the solution on the fine mesh but not in DMDA order
    PetscCall(VecCreate(PETSC_COMM_WORLD, &uf));
    PetscCall(VecSetSizes(uf, PETSC_DECIDE, 
        (ctx.piecex + 1) * (ctx.piecey + 1) * (ctx.piecez + 1)));
    PetscCall(VecSetType(uf, VECMPI));
    PetscCall(VecSetFromOptions(uf));
    PetscCall(VecSetUp(uf));
    PetscCall(VecDuplicate(uf, &ufcopy));

    PetscCall(DMCreateLocalVector(ctx.dmcoarse_node, &uclocal));
    PetscCall(DMGlobalToLocalBegin(ctx.dmcoarse_node, uc, INSERT_VALUES, uclocal));
    PetscCall(DMGlobalToLocalEnd(ctx.dmcoarse_node, uc, INSERT_VALUES, uclocal));
    PetscCall(DMDAVecGetArrayRead(ctx.dmcoarse_node, uclocal, &ucarray));
    for (PetscInt iz = zs; iz < zs + zm; iz++) {
    for (PetscInt iy = ys; iy < ys + ym; iy++) {
    for (PetscInt ix = xs; ix < xs + xm; ix++) {
        if (ix == ctx.Nx || iy == ctx.Ny || iz == ctx.Nz) continue;
        PetscInt globalindex = iz*ctx.Nx*ctx.Ny + ix*ctx.Ny + iy;
        Vec F, Fresult;
        PetscCall(VecCreateSeq(PETSC_COMM_SELF, 8, &F));
        PetscCall(VecSetFromOptions(F));
        PetscCall(VecSetUp(F));
        PetscCall(VecCreateSeq(PETSC_COMM_SELF, 
            (ctx.nx + 1) * (ctx.ny + 1) * (ctx.nz + 1), &Fresult));
        PetscCall(VecSetFromOptions(Fresult));
        PetscCall(VecSetUp(Fresult));
        MatStencil rowcol[8];
        rowcol[0].i = ix; rowcol[0].j = iy; rowcol[0].k = iz;
        rowcol[1].i = ix; rowcol[1].j = iy + 1; rowcol[1].k = iz;
        rowcol[2].i = ix + 1; rowcol[2].j = iy; rowcol[2].k = iz;
        rowcol[3].i = ix + 1; rowcol[3].j = iy + 1; rowcol[3].k = iz;
        rowcol[4].i = ix; rowcol[4].j = iy; rowcol[4].k = iz + 1;
        rowcol[5].i = ix; rowcol[5].j = iy + 1; rowcol[5].k = iz + 1;
        rowcol[6].i = ix + 1; rowcol[6].j = iy; rowcol[6].k = iz + 1;
        rowcol[7].i = ix + 1; rowcol[7].j = iy + 1; rowcol[7].k = iz + 1;
        PetscScalar ucval;
        for (PetscInt l = 0; l < 8; ++l) 
        {
            ucval = ucarray[rowcol[l].k][rowcol[l].j][rowcol[l].i];
            PetscCall(VecSetValues(F, 1, &l, &ucval, INSERT_VALUES));
        }
        PetscCall(VecAssemblyBegin(F));
        PetscCall(VecAssemblyEnd(F));
        PetscCall(MatMult(RR_local[globalindex], F, Fresult));
        PetscScalar *fresult;
        PetscCall(VecGetArray(Fresult, &fresult));
        PetscInt *fineindex = new PetscInt [(ctx.nx + 1) * (ctx.ny + 1) * (ctx.nz + 1)];
        PetscCall(projectionc2f(ix, iy, iz, &ctx, fineindex));
        PetscCall(VecSetValues(uf, (ctx.nx + 1) * (ctx.ny + 1) * (ctx.nz + 1), 
            fineindex, fresult, INSERT_VALUES));
        PetscCall(VecRestoreArray(Fresult, &fresult));

        delete [] fineindex;
        PetscCall(VecDestroy(&F));
        PetscCall(VecDestroy(&Fresult));
    }}}
    PetscCall(DMDAVecRestoreArrayRead(ctx.dmcoarse_node, uclocal, &ucarray));
    PetscCall(VecAssemblyBegin(uf));
    PetscCall(VecAssemblyEnd(uf));

    PetscCall(VecCopy(uf, ufcopy));
    PetscCall(VecAXPY(ufcopy, -1.0, uef));
    PetscCall(VecNorm(ufcopy, NORM_2, &diffnormf));
    PetscTime(&t5);
    ctx.tproject = t5 - t4;
    if (ctx.use_hdf5 == 1)
    {
    PetscViewer viewer;
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
    PetscCall(PetscObjectSetName((PetscObject)uf, "solution"));
    PetscCall(VecView(uf, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Save solution file successfully."));
    }
    PetscTime(&ctx.t2);
    PetscPrintf(PETSC_COMM_WORLD, "Time for setup: "
        "%g seconds\n", ctx.tsetup);
    PetscPrintf(PETSC_COMM_WORLD, "Time for coarse mesh assembling"
        " (include solving local fem): "
        "%g seconds (%g seconds)\n", ctx.tloop, ctx.tsolvelocal);
    
    PetscPrintf(PETSC_COMM_WORLD,
        "Time for solving coarse mesh MsFEM: %g seconds\n", ctx.tsolvecoarse);
    PetscPrintf(PETSC_COMM_WORLD, "Time for projection: %g seconds\n", ctx.tproject);
    PetscPrintf(PETSC_COMM_WORLD,
        "Relative error of MsFEM solution in coarse mesh: %g.\n", diffnormc / uecnorm);
    PetscPrintf(PETSC_COMM_WORLD,
        "Relative error of MsFEM solution in fine mesh: %g.\n", diffnormf / uefnorm);
    PetscPrintf(PETSC_COMM_WORLD, 
        "Time total: %g seconds\n", ctx.t2 - ctx.t1);

    PetscCall(MatDestroy(&Ac));
    PetscCall(VecDestroy(&bc));
    PetscCall(VecDestroy(&bclocal));
    PetscCall(DMDestroy(&ctx.dmcoarse_node));
    PetscCall(MatDestroy(&ctx.pnbd));
    PetscCall(VecDestroy(&ctx.zerobdcoarseDMDA));
    PetscCall(VecDestroy(&uec));
    PetscCall(VecDestroy(&uc));
    PetscCall(VecDestroy(&uccopy));
    PetscCall(VecDestroy(&uef));
    PetscCall(VecDestroy(&uf));
    PetscCall(VecDestroy(&ufcopy));
    PetscCall(VecDestroy(&uclocal));
    PetscCall(KSPDestroy(&ksp));
    for (PetscInt iz = zs; iz < zs + zm; iz++) {
    for (PetscInt iy = ys; iy < ys + ym; iy++) {
    for (PetscInt ix = xs; ix < xs + xm; ix++) {
        if (ix == ctx.Nx || iy == ctx.Ny || iz == ctx.Nz) continue;
        PetscInt globalindex = iz*ctx.Nx*ctx.Ny + ix*ctx.Ny + iy;
        PetscCall(MatDestroy(&RR_local[globalindex]));
    }}}
    delete[] zerorows;
    PetscCall(PetscFinalize());
    return 0;
}