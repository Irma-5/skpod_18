#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
double *epss;
int itmax = 100;
int i,j,k;
double w = 0.5;
double A [N][N][N];

void relax();
void init();
void verify();

int main(int an, char **as)
{
        int it, ind;
        double start = omp_get_wtime();
        int thrds = omp_get_max_threads();
        double * epss = malloc(thrds * sizeof(double)); 

    #pragma omp parallel shared(A, epss)
    { 
        init(A);
    #pragma omp single
        for(it=1; it<=itmax; it++)
        {
                double eps = 0.;

                for (ind=0; ind<thrds; ind++) epss[ind]=0.0;
                relax(A, epss, w, thrds);
                for (ind=0; ind<thrds; ind++)
                {
                    eps = Max(eps, epss[ind]);
                }
                printf("it=%4i   eps=%f\n", it, eps);
                if (eps < maxeps) break;
        }
    }
	verify(A);
        double end = omp_get_wtime();
        printf("Time: %f", end-start);
        return 0;
}

void init(double A[N][N][N])
{
        for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
        for(k=0; k<=N-1; k++)
        {

            if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)     A[i][j][k]= 0.;
            else A[i][j][k]= ( 4. + i + j + k) ;
        }
}

void relax(double A[N][N][N], double * epss, double w, int thrds)
{
    int i, j, k, num, tmp, tmp_i, N_i;
    int iters = (N-2)*(N-2)/thrds;
    int len = (N-2)*(N-2);

    for (tmp=0; tmp < len; tmp += iters)
    {
        if ((tmp+iters) < len) N_i = tmp+iters;
        else N_i = len;

        #pragma omp task private(i, j, k, tmp_i, num) firstprivate (tmp, N_i)
        {
            num = omp_get_thread_num();
            for(tmp_i=tmp; tmp_i < N_i; tmp_i++)
        {
            i = 1 + tmp_i / (N-2);
            j = 1+ (tmp_i % (N-2));

            for(k=1+(i+j)%2; k<=N-2; k+=2)
            {
                double b;
                b = w*((A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]
                +A[i][j][k-1]+A[i][j][k+1] )/6. - A[i][j][k]);

                epss[num] =  Max(fabs(b), epss[num]);
                A[i][j][k] = A[i][j][k] + b;
            }
        }

        }
    }
    #pragma omp taskwait

    for (tmp=0; tmp < len; tmp += iters)
    {

        if ((tmp+iters) < len) N_i = tmp+iters;
        else N_i = len;

        #pragma omp task private(i, j, k, tmp_i) firstprivate(tmp, N_i)
        {
            for (tmp_i=tmp; tmp_i<N_i; tmp_i++)
        {
            i = 1 + tmp_i / (N-2);
            j = 1 + (tmp_i % (N-2));

            for (k=1+(i+j+1)%2; k<=N-2;  k+=2)
            {
                double b;
                b = w*((A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]
                +A[i][j][k-1]+A[i][j][k+1])/6. - A[i][j][k]);
                A[i][j][k] = A[i][j][k] + b;
            }
        }
        }

    }
    #pragma omp taskwait

}

void verify(double A[N][N][N])
{
        double s;

        s=0.;

        for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
        for(k=0; k<=N-1; k++)
        {
                s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
        }
        printf("  S = %f\n",s);

}
