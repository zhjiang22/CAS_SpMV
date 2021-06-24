#include<iostream>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <hip/hip_runtime.h>

enum sparse_operation {operation_none=0,operation_transpose=1} ;


__global__ void spmv_pcsr_kernel1(double * d_val,double * d_vector,int * d_cols,int d_nnz, double * d_v)
{
    	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    	int icr = blockDim.x * gridDim.x;
    	while (tid < d_nnz){
		d_v[tid] = d_val[tid] * d_vector[d_cols[tid]];
        	tid += icr;
    	}
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