#include<iostream>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <hip/hip_runtime.h>

enum sparse_operation {operation_none=0,operation_transpose=1} ;


//__launch_bounds__(VECTORS_PER_BLOCK * THREADS_PER_VECTOR,1)
template <int VECTOR_PER_BLOCK, int THREADS_PER_VECTOR>
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
	__shared__ volatile int space[VECTOR_PER_BLOCK][2];
	int i, row;
	double sum;
	const int THREADS_PER_BLOCK = VECTOR_PER_BLOCK * THREADS_PER_VECTOR;
	const int thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
	int laneId = threadIdx.x & (THREADS_PER_VECTOR - 1);       //lane index in the vector
	int vector_id = thread_id / THREADS_PER_VECTOR;
	int vector_lane = threadIdx.x / THREADS_PER_VECTOR;     //vector index in the thread block
	int num_vectors = VECTOR_PER_BLOCK * gridDim.x;
  	// Get the row index
  	// Broadcast the value to other threads in the same warp and compute the row index of each vector
  	for (int row = vector_id; row < m; row += num_vectors){
// Use two threads to fetch the row offset
	    if (laneId < 2) space[vector_lane][laneId] = rowptr[row + laneId];
		const int rowStart = space[vector_lane][0];
		const int rowEnd = space[vector_lane][1];
		sum = 0.0;
// Compute dot product
  			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
    			sum += value[i] * x[colindex[i]];
			}
    // Intra-vector reduction
#pragma unroll
    	for (i = (THREADS_PER_VECTOR >> 1); i > 0; i >>= 1) {
        	sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
		}
    // Save the results
    	if (laneId == 0) y[row] = alpha * sum + beta * y[row];
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
	int NonZ;
	hipMemcpy((void *)&NonZ, (void *)(hrowptr + hm), sizeof(int), hipMemcpyDeviceToHost);
	hipDeviceSynchronize();
	const int block_dim = 256;
	double mean_elements = (double)(NonZ) / hm;
	if (mean_elements <= 2) {
		const int temp = 2;
		const int vecs = block_dim / temp;
		int grid_dim = (hm + (vecs) - 1) / ((vecs));
    	device_sparse_spmv<vecs, temp><<<grid_dim, block_dim>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);
	} else if (mean_elements <= 4) {
		const int temp = 4;
		const int vecs = block_dim / temp;
		int grid_dim = (hm + (vecs) - 1) / ((vecs));
    	device_sparse_spmv<vecs, temp><<<grid_dim, block_dim>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);
	} else if (mean_elements <= 64) {
		const int temp = 8;
		const int vecs = block_dim / temp;
		int grid_dim = (hm + (vecs) - 1) / ((vecs));
    	device_sparse_spmv<vecs, temp><<<grid_dim, block_dim>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);
	} else if (mean_elements < 256) {
		const int temp = 16;
		const int vecs = block_dim / temp;
		int grid_dim = (hm + (vecs) - 1) / ((vecs));
    	device_sparse_spmv<vecs, temp><<<grid_dim, block_dim>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);
	} else {
		const int temp = 32;
		const int vecs = block_dim / temp;
		int grid_dim = (hm + (vecs) - 1) / ((vecs));
    	device_sparse_spmv<vecs, temp><<<grid_dim, block_dim>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);		
	}

}