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
								   int THREADS_PER_VECTOR,
								   int *cudaRowCounter)
{
	int i, row, rowStart, rowEnd;
	double sum;
	int laneId = threadIdx.x % THREADS_PER_VECTOR; //lane index in the vector
	int vectorId = threadIdx.x / THREADS_PER_VECTOR; //vector index in the thread block
	int warpLaneId = threadIdx.x & (warpSize - 1); //lane index in the warp
	int warpVectorId = warpLaneId / THREADS_PER_VECTOR; //vector index in the warp
	extern __shared__ volatile int space[][2];
  	// Get the row index
  	if (warpLaneId == 0)
    	row = atomicAdd(cudaRowCounter, warpSize / THREADS_PER_VECTOR);
  	// Broadcast the value to other threads in the same warp and compute the row index of each vector
  	row = __shfl(row, 0) + warpVectorId;
  	while (row < m) {
// Use two threads to fetch the row offset
	    if (laneId < 2) space[vectorId][laneId] = rowptr[row + laneId];
		rowStart = space[vectorId][0];	rowEnd = space[vectorId][1];
		sum = 0.0;
// Compute dot product
		/*if (THREADS_PER_VECTOR == warpSize) {
  // Ensure aligned memory access
  			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;
  // Process the unaligned part
  			if (i >= rowStart && i < rowEnd) {
    			sum += value[i] * x[colindex[i]];
			}
    // Process the aligned part
  			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
    			sum += value[i] * x[colindex[i]];
			}
		} else {
  			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
    			sum += value[i] * x[colindex[i]];
			}
 		}*/
		 for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
    			sum += value[i] * x[colindex[i]];
			}
    // Intra-vector reduction
    	for (i = (THREADS_PER_VECTOR >> 1); i > 0; i >>= 1) {
        	sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
		}
    // Save the results
    	if (laneId == 0) y[row] = alpha * sum + beta * y[row];
    // Get a new row index
    	if(warpLaneId == 0)
       		row = atomicAdd(cudaRowCounter, warpSize / THREADS_PER_VECTOR);
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
	int NonZ;
	hipMemcpy((void *)&NonZ, (void *)(hrowptr + hm), sizeof(int), hipMemcpyDeviceToHost);
	hipDeviceSynchronize();
	double mean_elements = (double)(NonZ) / hm;
	temp = 2;
	int block_dim = 1024;
	int grid_dim = (hm + block_dim - 1) / (block_dim / temp);
    device_sparse_spmv<<<grid_dim, block_dim, block_dim / temp * sizeof(double)>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy, temp, cudaRowCounter);
}