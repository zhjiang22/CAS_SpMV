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
                     			   double* y,
								   const int power,
								   int *cudaRowCounter)
{
	int row;
	int THREADS_PER_VECTOR = (1 << power);
	int laneId = threadIdx.x & (THREADS_PER_VECTOR - 1) ; //lane index in the vector
	int vectorId = (threadIdx.x >> power); //vector index in the thread block
	int warpLaneId = threadIdx.x & (warpSize - 1); //lane index in the warp
	int warpVectorId = (warpLaneId >> power); //vector index in the warp
	extern __shared__ volatile int space[][2];
  	// Get the row index
  	if (warpLaneId == 0)
    	 row = atomicAdd(cudaRowCounter, (warpSize >> power));
  	// Broadcast the value to other threads in the same warp and compute the row index of each vector
  	    row = __shfl(row, 0) + warpVectorId;
  	while (row < m) {
// Use two threads to fetch the row offset
	    if (laneId < 2) space[vectorId][laneId] = rowptr[row + laneId];
		int rowStart = space[vectorId][0];	int rowEnd = space[vectorId][1];
		double sum = 0.0;
		 for (int i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR)
    			sum += value[i] * x[colindex[i]];
    // Intra-vector reduction
        sum += __shfl_down(sum, 2, THREADS_PER_VECTOR);
        sum += __shfl_down(sum, 1, THREADS_PER_VECTOR);
    // Save the results
    	if (laneId == 0) y[row] = alpha * sum + beta * y[row];
    // Get a new row index
    	if(warpLaneId == 0) row = atomicAdd(cudaRowCounter, (warpSize >> power));
    // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
       	row = __shfl(row, 0) + warpVectorId;
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
	
	temp = 2; // the power of 2
	int vectorLen = (1 << temp);
	int block_dim = 256;
	int grid_dim = (hm + block_dim - 1) / (block_dim);
    device_sparse_spmv<<<grid_dim, block_dim, block_dim / vectorLen * sizeof(double)>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy, temp, cudaRowCounter);
}
