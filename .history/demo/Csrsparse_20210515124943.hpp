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
        for (int j = rowptr[idx]; j < rowptr[idx + 1]; j++)
			y0 += value[j]*x[colindex[j]];
		y[idx] = alpha * y0 + beta * y[idx];
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
        device_sparse_spmv<<<1,256>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);
}