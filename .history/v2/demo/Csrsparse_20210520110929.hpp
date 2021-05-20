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
	int lane = t & 31;
	int warpsPerBlock = blockDim.x / 32;
	int row = (blockIdx.x * warpsPerBlock) + (tid / warpSize);
	__shared__ volatile T result[BlockDim];
    if (row < N) {
      	int rowStart = rowptr[row], rowEnd = rowptr[row+1];
      	double sum = 0.0;
      	for (int i = rowStart + lane; i < rowEnd; i += warpSize)
         	sum += value[i] * x[colindex[i]];
		reslut[t] = sum;
		__syncthreads();
		if (lane < 16) reslut[t] += reslut[t + 16];
		if (lane < 8)  reslut[t] += reslut[t + 8];
		if (lane < 4)  reslut[t] += reslut[t + 4];
		if (lane < 2)  reslut[t] += reslut[t + 2];
		if (lane < 1)  reslut[t] += reslut[t + 1];
		__syncthreads();
    } 
	if (lane == 0)	d_out[row] = reslut[t];
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
	int grid_dim = (hm + block_dim - 1) / (block_dim / 32);
    device_sparse_spmv<<<grid_dim, block_dim>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);
}