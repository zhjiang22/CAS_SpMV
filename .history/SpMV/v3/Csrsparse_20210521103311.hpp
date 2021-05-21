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
	int i, row, rowStart, rowEnd;
	doube sum;
	int laneId = threadIdx.x % THREADS_PER_VECTOR; //lane index in the vector
	int vectorId = threadIdx.x / THREADS_PER_VECTOR; //vector index in the thread block
	int warpLaneId = threadIdx.x & (warpSize - 1); //lane index in the warp
	int warpVectorId = warpLaneId / THREADS_PER_VECTOR; //vector index in the warp
	__shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];
  	// Get the row index
  	if (warpLaneId == 0)
    	row = atomicAdd(cudaRowCounter, warpSize / THREADS_PER_VECTOR);
  	// Broadcast the value to other threads in the same warp and compute the row index of each vector
  	row = __shfl(row, 0) + warpVectorId;
  	while (row < N) {
// Use two threads to fetch the row offset
		if (laneId < 2) {
  			space[vectorId][laneId] = d_ptr[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];
		sum = 0;
// Compute dot product
		if (THREADS_PER_VECTOR == warpSize) {
  // Ensure aligned memory access
  			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;
  // Process the unaligned part
  			if (i >= rowStart && i < rowEnd) {
    			sum += d_val[i] * d_vector[d_cols[i]];
    // Process the aligned part
  			for (i += THREADS_PER_VECTOR; i <
      rowEnd; i += THREADS_PER_VECTOR) {
    sum += d_val[i] *
        d_vector[d_cols[i]];
}
} else {
  for (i = rowStart + laneId; i <
      rowEnd; i +=
       THREADS_PER_VECTOR) {
    sum += d_val[i] *
        d_vector[d_cols[i]];
}
 } }
    // Intra-vector reduction
    for (i = THREADS_PER_VECTOR >> 1; i > 0;
        i >>= 1) {
         sum += __shfl_down(sum, i,
             THREADS_PER_VECTOR);
}
    // Save the results
    if (laneId == 0) {
       d_out[row] = sum;
}
    // Get a new row index
    if(warpLaneId == 0){
       row = atomicAdd(cudaRowCounter, warpSize /
          THREADS_PER_VECTOR);
}
    // Broadcast the row index to the other
        threads in the same warp and compute
        the row index of each vector
       row = __shfl(row, 0) + warpVectorId;
} 
}

template <typename T, int
    THREADS_PER_VECTOR, int
    MAX_NUM_VECTORS_PER_BLOCK>
__global__ void spmv_light_kernel(int*
    cudaRowCounter, int* d_ptr, int*
    d_cols,T* d_val, T* d_vector, T*
    d_out,int N) {
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