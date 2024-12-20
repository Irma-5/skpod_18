#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double w = 0.5;
double eps;

double A [N][N][N];

void relax();
void init();
void verify();

int main(int an, char **as)
{
        int it;
        double t0 = omp_get_wtime();

        init();

        for(it=1; it<=itmax; it++)
        {
                eps = 0.;
                relax();
                printf( "it=%4i   eps=%f\n", it,eps);
                if (eps < maxeps) break;
        }

        verify();
        double t1 = omp_get_wtime();
        printf("Time: %f", (t1-t0));

        return 0;
}


void init()
{
        for(k=0; k<=N-1; k++)
        for(j=0; j<=N-1; j++)
        for(i=0; i<=N-1; i++)
        {
                if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)     A[i][j][k]= 0.;
                else A[i][j][k]= ( 4. + i + j + k) ;
        }
}


void relax()
{

        for(k=1; k<=N-2; k++)
        for(j=1; j<=N-2; j++)
        for(i=1+(k+j)%2; i<=N-2; i+=2)
        {
                double b;
                b = w*((A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]
                +A[i][j][k-1]+A[i][j][k+1] )/6. - A[i][j][k]);
                eps =  Max(fabs(b),eps);
                A[i][j][k] = A[i][j][k] + b;
        }

        for(k=1; k<=N-2; k++)
        for(j=1; j<=N-2; j++)
        for(i=1+(k+j+1)%2; i<=N-2; i+=2)
        {
                double b;
                b = w*((A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]
                +A[i][j][k-1]+A[i][j][k+1] )/6. - A[i][j][k]);
                A[i][j][k] = A[i][j][k] + b;
        }

}

void verify()
{
        double s;

        s=0.;
        for(k=0; k<=N-1; k++)
        for(j=0; j<=N-1; j++)
        for(i=0; i<=N-1; i++)
        {
                s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
        }
        printf("  S = %f\n",s);

}

