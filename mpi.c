#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (2*2*2*2*2*2 + 2)
#define TAG_PASS_FIRST 100
#define TAG_PASS_LAST 200
#define N2 (N * N)
#define N3 (N * N2)

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double w = 0.5;
double eps;
double b, s = 0.;
double A[N][N][N];

void relax();
void init();
void verify();

void change_first();
void change_last();
void waitAll();

int size, rank, left_idx, right_idx, step;

MPI_Request req_buf[4];
MPI_Status stat_buf[4];

int main(int an, char **as) {
    int it;

    MPI_Init(&an, &as);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size_block = (N - 2) / size;
    int ost = (N - 2) % size;

    lst_r = 1;
    for (i = 0; i < rank + 1; i++) {
        lst_r += (size_block + (i < ost ? 1 : 0));
    }
    left_idx = right_idx - (size_block + (rank < ost ? 1 : 0));
    step = right_idx - left_idx;

    double time_start, time_end;
    time_start = MPI_Wtime();

    init();

    for (it = 1; it <= itmax; it++) {
        eps = 0.;
        relax();
        if (!rank)
           printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps)
            break;
    }

    MPI_Gather(A[left_idx], step * N2, MPI_DOUBLE, A[1], step * N2,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

    verify();
    time_end = MPI_Wtime();

    if (!rank)
        printf("Elapsed time: %lf.\n", time_end - time_start);

    MPI_Finalize();

    return 0;
}

void init() {
    for (i = left_idx; i < right_idx; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                if (j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.;
                else
                    A[i][j][k] = (4. + i + j + k);
            }
}

void relax() {

    double eps_local = 0.;

   change_last();
   change_first();
   waitAll();

    for (i = left_idx; i < right_idx; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1 + (i + j) % 2; k <= N - 2; k += 2) {
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                          A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
                eps_local = Max(fabs(b), eps_local);
                A[i][j][k] = A[i][j][k] + b;
            }

    change_last();
    change_first();
    waitAll();

    for (i = left_idx; i < right_idx; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1 + (i + j + 1) % 2; k <= N - 2; k += 2) {
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                          A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
                A[i][j][k] = A[i][j][k] + b;
            }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&eps_local, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

}

void verify() {
    double s_local = 0.;

    for (i = left_idx; i < right_idx; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                s_local = s_local + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N3);
            }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&s_local, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("  S = %f\n", s);
    }
}

void change_last() {
    if (rank)
        MPI_Irecv(A[left_idx - 1], N2, MPI_DOUBLE, rank - 1, TAG_PASS_LAST, MPI_COMM_WORLD, &req_buf[0]);
    if (rank != size - 1)
        MPI_Isend(A[right_idx - 1], N2, MPI_DOUBLE, rank + 1, TAG_PASS_LAST, MPI_COMM_WORLD, &req_buf[2]);
}

void change_first() {
    if (rank != size - 1)
        MPI_Irecv(A[right_idx], N2, MPI_DOUBLE, rank + 1, TAG_PASS_FIRST, MPI_COMM_WORLD, &req_buf[3]);
    if (rank)
        MPI_Isend(A[left_idx], N2, MPI_DOUBLE, rank - 1, TAG_PASS_FIRST, MPI_COMM_WORLD,  &req_buf[1]);
}

void waitAll() {
    int count = 4, shift = 0;
    if (!rank) {
        count -= 2;
        shift = 2;
    }
    if (rank == size - 1) {
        count -= 2;
    }

    MPI_Waitall(count, &req_buf[shift], &stat_buf[0]);

}
