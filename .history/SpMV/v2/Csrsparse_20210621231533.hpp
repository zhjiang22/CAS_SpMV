#include<iostream>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <hip/hip_runtime.h>

enum sparse_operation {operation_none=0,operation_transpose=1} ;


__global__ void device_sparse_spmv(int trans,
                                   const int alpha,
	                               const int beta,
                     			   int m,
                     			   int n,
               					   const int* rowptr,
	    						   const int* colindex,
	       						   const double* value,
               					   const double*  x,
                     			   double* y)
{
	int tid = threadIdx.x;
	int lane = tid & (warpSize - 1);
	int warpsPerBlock = blockDim.x / warpSize;
	int row = (blockIdx.x * warpsPerBlock) + (tid / warpSize);
	double sum = 0.0;
    if (row < m) {
      	int rowStart = rowptr[row], rowEnd = rowptr[row + 1];
      	double sum = 0.0;
      	for (int i = rowStart + lane; i < rowEnd; i += warpSize)
         	sum += value[i] * x[colindex[i]];
		sum *= alpha;
    } 

	volatile double *res = result;
	for (int step = (warpSize / 2); step > 0; step >>= 1) {
		if (lane < step) res[tid] += res[tid + step];
		//__syncthreads();
	}
	if (row < m && lane == 0)
		y[row] = res[tid] + beta * y[row];
}



void  sparse_spmv(int htrans,
                  const int halpha,
	       		  const int hbeta,
                  int hm,
                  int hn,
                  const int* hrowptr,
	      		  const int* hcolindex,
	       		  const double* hvalue,
               	  const double* hx,
                  double* hy)
{
	int block_dim = 1024;
	int grid_dim = (hm + (block_dim / 64) - 1) / (block_dim / 64);
    device_sparse_spmv<<<grid_dim, block_dim, block_dim * sizeof(double)>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);
}