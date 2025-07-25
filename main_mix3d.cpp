#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>
#include <petscksp.h>
#include <mpi.h>
PetscScalar kfunc(PetscScalar x, PetscScalar y, PetscScalar z) {
    // Example function for k
    return 2.0 + x*y*z;
}
int main(int argc, char **argv) {
    PetscMPIInt rank, size;
    PetscLogDouble t1, t2, t3, t4;
    PetscInt Istart, Iend;
    PetscInt save = 0;

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MPI Size: %d\n", size));
    PetscTime(&t1);
    PetscInt n = 40;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "3D Dancy problem with n = %ld\n", n));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-save", &save, NULL));
    PetscScalar diagA = 1.0/2/n/n/n;
    PetscScalar mass[6];
    for (PetscInt i = 0; i < 6; i++) {mass[i] = diagA;}
    PetscScalar dp[6] = {-1.0*n, 1.0*n , -1.0*n, 
        1.0*n, -1.0*n, 1.0*n};
    Mat B, BT, A;
    Vec M, Minv, Mone, F, u;
    PetscInt facenum = 3*n*n*(n+1);
    PetscCall(VecCreate(PETSC_COMM_WORLD, &M));
    PetscCall(VecSetSizes(M, PETSC_DECIDE, facenum));
    PetscCall(VecSetType(M, VECMPI));
    PetscCall(VecSetFromOptions(M));
    PetscCall(VecSetUp(M));
    PetscCall(VecDuplicate(M, &Minv));
    PetscCall(VecDuplicate(M, &Mone));
    PetscCall(VecSet(M, 0.0));
    PetscCall(VecSet(Minv, 0.0));
    PetscCall(VecSet(Mone, 1.0));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &F));
    PetscCall(VecSetSizes(F, PETSC_DECIDE, n*n*n));
    PetscCall(VecSetType(F, VECMPI));
    PetscCall(VecSetFromOptions(F));
    PetscCall(VecSetUp(F));
    PetscCall(VecDuplicate(F, &u));
    PetscCall(VecSet(F, 1.0));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
    PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, n*n*n, facenum));
    PetscCall(MatSetType(B, MATMPIAIJ));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatSetUp(B));
    PetscCall(MatMPIAIJSetPreallocation(B, 6, NULL, 6, NULL));
    PetscCall(MatGetOwnershipRange(B, &Istart, &Iend));
    for (PetscInt i = Istart; i < Iend; i++) {
        PetscInt iz = i / (n * n);
        PetscInt ires = i - iz * n * n;
        PetscInt ix = ires / n;
        PetscInt iy = ires - ix * n;
        PetscInt row = i;
        PetscInt col[6];
        col[0] = iz*n*(n+1) + ix*(n+1) + iy;
        col[1] = col[0] + 1;
        col[2] = n*n*(n+1) + iz*n*(n+1) + ix*n + iy;
        col[3] = col[2] + n;
        col[4] = 2*n*n*(n+1) + iz*n*n + ix*n + iy;
        col[5] = col[4] + n*n;
        PetscCall(MatSetValues(B, 1, &row, 6, col, dp, ADD_VALUES));
        PetscScalar centx = (PetscScalar)(ix + 0.5) / n;
        PetscScalar centy = (PetscScalar)(iy + 0.5) / n;
        PetscScalar centz = (PetscScalar)(iz + 0.5) / n;
        PetscScalar kinv = 1.0/kfunc(centx, centy, centz);
        PetscScalar val[6];
        for (PetscInt l = 0; l < 6; l++) {val[l] = mass[l] * kinv;}
        PetscCall(VecSetValues(M, 6, col, val, ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(M));
    PetscCall(VecAssemblyEnd(M));
    PetscCall(VecPointwiseDivide(Minv, Mone, M));
    PetscCall(MatTranspose(B, MAT_INITIAL_MATRIX, &BT));
    PetscCall(MatDiagonalScale(B, NULL, Minv));
    PetscCall(MatMatMult(B, BT, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A));
    PetscCall(PetscTime(&t2));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, 
        "Matrix assembly time: %g seconds\n", (t2 - t1)));

    KSP ksp;
    PC pc;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMG));
    PetscCall(KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 500));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, F, u));
    PetscCall(PetscTime(&t3));
    // PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD));
    PetscInt its;
    KSPGetIterationNumber(ksp, &its);
    PetscPrintf(PETSC_COMM_WORLD, "Iterations to convergence: %ld\n", its);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "KSP Solve time: %g seconds\n", 
    (t3 - t2)));
    PetscReal norm;
    PetscCall(VecNorm(u, NORM_2, &norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solution norm: %g\n", (double)norm));
    if (save) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Saving solution to HDF5...\n"));
        PetscViewer viewer;
        PetscCall(PetscObjectSetName((PetscObject)u, "solution"));
        PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, 
            "solution.h5", FILE_MODE_WRITE, &viewer));
        PetscCall(VecView(u, viewer));
        PetscCall(PetscViewerDestroy(&viewer));
    }
    PetscCall(PetscTime(&t4));
    if (save) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Save time: %g seconds\n", 
        (t4 - t3)));}
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total time: %g seconds\n", (t4 - t1)));

    PetscCall(VecDestroy(&M));
    PetscCall(VecDestroy(&Minv));
    PetscCall(VecDestroy(&Mone));
    PetscCall(VecDestroy(&F));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&BT));
    PetscCall(MatDestroy(&B));
    PetscCall(VecDestroy(&u));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Finalizing PETSc...\n"));

    PetscCall(PetscFinalize());

    return 0;
}