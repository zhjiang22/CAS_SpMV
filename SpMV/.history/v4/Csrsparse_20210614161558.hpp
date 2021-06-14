#include<iostream>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <hip/hip_runtime.h>

#define MIN((a), (b)) ((a) < (b) ? (a) : (b))
#define MIN((a), (b)) ((a) > (b) ? (a) : (b))

enum sparse_operation {operation_none=0,operation_transpose=1} ;

__device__ void BinarySearch(int start_k, int end_k, const int* rowptr, int m, int NonZ,
							 int* begin_x, int *begin_y, int *end_x, int *end_y) {
    int x_min = MAX(start_k - NonZ, 0), x_max = MIN(start_k, m);
	while (x_min < x_max) {
		int mid = (x_min + x_max) >> 1;
		if (rowptr[mid] < start_k - mid - 1)
			x_min = mid + 1;
		else x_max = mid;
	}
	*begin_x = MIN(x_min, m); *begin_y = start_k - x_min;

    x_min = MAX(end_k - NonZ, 0), x_max = MIN(end_k , m);
	while (x_min < x_max) {
		int mid = (x_min + x_max) >> 1;
		if (rowptr[mid] < end_k - mid - 1)
			x_min = mid + 1;
		else x_max = mid;
	}
	*end_x = MIN(x_min, m); *end_y = end_k - x_min;
	
}

__global__ void device_sparse_spmv(int trans,
                                   const int alpha,
	                               const int beta,
                     			   int m,
                     			   int n,
               					   const int* rowptr,
	    						   const int* colindex,
	       						   const double* value,
               					   const double*  x,
                     			   double* y,
								   int THREADS_PER_VECTOR,
								   int *cudaRowCounter)
{
	int tid = threadIdx.x;
	int start_k = MIN(items_per_thread * tid,  m + NonZ);
	int end_k = MIN(start_k + items_per_thread, m + NonZ);
	int begin_x, begin_y, end_x, end_y;
	BinarySearch(start_k, end_k, rowptr, m, NonZ, &begin_x, &begin_y, &end_x, &end_y);
	double sum;
	for (; begin_x < end_x; ++begin_x) {
		sum = 0.0;
		for (; begin_y < rowptr[begin_x]; ++begin_y)
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
	int *cudaRowCounter;
	int temp = 0;
	hipMalloc((void **)&cudaRowCounter, sizeof(int));
	hipMemcpy((void *)cudaRowCounter, (void *)&temp, sizeof(int), hipMemcpyHostToDevice);
	double mean_elements = (double)(hrowptr[hm]) / hm;
	if (mean_elements <= 2) temp = 2;
	else if (mean_elements <= 4) temp = 4;
	else if (mean_elements <= 64) temp = 8;
	else temp = 16;
	int block_dim = 1024;
	int grid_dim = (hm + block_dim - 1) / (block_dim);
    device_sparse_spmv<<<grid_dim, block_dim, block_dim * sizeof(double)>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy, temp, cudaRowCounter);
}