#include<iostream>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <hip/hip_runtime.h>

enum sparse_operation {operation_none=0,operation_transpose=1} ;

__device__ void BinarySearch(int start_k, int end_k, const int* rowptr, int m, int NonZ,
							 int* begin_x, int *begin_y, int *end_x, int *end_y) {
    int x_min = (start_k - NonZ) > 0 ? (start_k - NonZ) : 0;
	int x_max = start_k < m ? start_k : m;
	while (x_min < x_max) {
		int mid = (x_min + x_max) >> 1;
		if (rowptr[mid] < start_k - mid - 1)
			x_min = mid + 1;
		else x_max = mid;
	}
	*begin_x = x_min < m ? x_min : m;
	*begin_y = start_k - x_min;

    x_min = end_k - NonZ > 0 ? end_k - NonZ : 0;
	x_max = end_k < m ? end_k : m;
	while (x_min < x_max) {
		int mid = (x_min + x_max) >> 1;
		if (rowptr[mid] < end_k - mid - 1)
			x_min = mid + 1;
		else x_max = mid;
	}
	*end_x = x_min < m ? x_min : m;
	*end_y = end_k - x_min;
}

__global__ void device_sparse_spmv(int trans,
                                   const int alpha,
	                               const int beta,
                     			   int m,
                     			   int n,
								   int NonZ,
								   int items_per_thread,
               					   const int* rowptr,
	    						   const int* colindex,
	       						   const double* value,
               					   const double*  x,
                     			   double* y)
{
	int tid = threadIdx.x;
	int start_k = (items_per_thread * tid) < (m + NonZ) ? (items_per_thread * tid) : (m + NonZ);
	int end_k = (start_k + items_per_thread) < (m + NonZ) ? (start_k + items_per_thread) : (m + NonZ);
	int begin_x, begin_y, end_x, end_y;
	BinarySearch(start_k, end_k, rowptr, m, NonZ, &begin_x, &begin_y, &end_x, &end_y);
	double sum;
	for (; begin_x < end_x; ++begin_x) {
		sum = 0.0;
		for (; begin_y < rowptr[begin_x]; ++begin_y)
			sum += value[begin_y] * x[colindex[begin_y]];
		y[begin_x] = alpha * sum + beta * y[begin_x];
	}
	
	sum = 0.0;
	for (; begin_y < end_y; ++begin_y)
		sum += value[begin_y] * x[colindex[begin_y]];
	atomicAdd(&y[end_x], alpha * sum + beta * y[end_x]);
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
	int NonZ;
	hipMemcpy(hrowptr + hm, )
	int sum_k = NonZ + hm;
	int grid_dim = (sum_k + (block_dim * 64) - 1) / (block_dim * 64);
	printf("sum_k: %d grid: %d NonZ: %d\n", sum_k, grid_dim, NonZ);
    device_sparse_spmv<<<grid_dim, block_dim, block_dim * sizeof(double)>>>(htrans,halpha,hbeta,hm,hn,NonZ,64, hrowptr,hcolindex,hvalue,hx,hy);
}