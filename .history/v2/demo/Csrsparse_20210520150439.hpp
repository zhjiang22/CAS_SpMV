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
	int lane = tid & (64 - 1);
	int warpsPerBlock = blockDim.x / 64;
	int row = (blockIdx.x * warpsPerBlock) + (tid / 64);
	__shared__ volatile double result[1024];
	result[tid] = 0.0;
	__syncthreads();
    if (row < m) {
      	int rowStart = rowptr[row], rowEnd = rowptr[row+1];
      	double sum = 0.0;
      	for (int i = rowStart + lane; i < rowEnd; i += 64)
         	sum += value[i] * x[colindex[i]];
		result[tid] = sum;
    } 
	__syncthreads();

	for (int step = 32; step >= 1; step >>= 1) {
		if (lane < step) result[tid] += result[tid + step];
		__syncthreads();
	}
	if (row < m && lane == 0)
		y[row] = result[tid];
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
    device_sparse_spmv<<<grid_dim, block_dim>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);
}