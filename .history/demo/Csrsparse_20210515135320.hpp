#include<iostream>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <hip/hip_runtime.h>

enum sparse_operation {operation_none=0,operation_transpose=1} ;


__global__ void device_sparse_spmv(int        trans,
               const int               alpha,
	       const int               beta,
                     int               m,
                     int               n,
               const int*              rowptr,
	       const int*              colindex,
	       const double*           value,
               const double*           x,
                     double*           y
			)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        double y0 = 0;
		if (idx < m) {
			for (int j = rowptr[idx]; j < rowptr[idx + 1]; j++)
				y0 += value[j]*x[colindex[j]];
			y[idx] = alpha * y0 + beta * y[idx];
		}
}


__global__ void device_sparse_spmv2(int        trans,
               const int               alpha,
	       const int               beta,
                     int               m,
                     int               n,
               const int*              rowptr,
	       const int*              colindex,
	       const double*           value,
               const double*           x,
                     double*           y
			)
{
	extern __shared__ double s_sum[];
	int tid = threadIdx.x;
	int gid = threadIdx.x + rowptr[blockIdx.x];
	int cnt_row = rowptr[blockIdx.x + 1] - rowptr[blockIdx.x];
	s_sum[tid] = 0.0;
	__syncthreads();

	if (tid < cnt_row)
		s_sum[tid] = value[gid] * x[colindex[gid]];
	__syncthreads();
	
	double tmp;
	for (int j = 1; j < blockDim.x; j *= 2) {
		if (tid - j >= 0)
			tmp = s_sum[tid - j];
		__syncthreads();
		if (tid - j >= 0)
			s_sum[tid] += tmp;
		__syncthreads();		
	}
	if (tid == blockDim.x - 1)
		y[blockIdx.x] = s_sum[tid];
}


void  sparse_spmv(int                  htrans,
               const int               halpha,
	       const int               hbeta,
                     int               hm,
                     int               hn,
               const int*              hrowptr,
	       const int*              hcolindex,
	       const double*           hvalue,
               const double*           hx,
                     double*           hy
			)
{
    device_sparse_spmv2<<<hm, hn>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);
}