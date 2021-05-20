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
	int lane = tid & (warpSize - 1);
	int warpsPerBlock = blockDim.x / warpSize;
	int row = (blockIdx.x * warpsPerBlock) + (tid / warpSize);
	__shared__ volatile double result[1024];
	result[tid] = 0.0;
	__syncthreads();
    if (row < m) {
      	int rowStart = rowptr[row], rowEnd = rowptr[row+1];
      	double sum = 0.0;
      	for (int i = rowStart + lane; i < rowEnd; i += warpSize)
         	sum += value[i] * x[colindex[i]];
		result[tid] = sum;
    } 
	__syncthreads();

	for (int step = (warpSize - 1); step >= 1; step >>= 1) {
		if (lane < step) result[tid] += result[tid + step];
		__syncthreads();
	}
	if (row < m && lane == 0)
		y[row] = result[tid];
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
	int grid_dim = (hm + (block_dim / 32) - 1) / (block_dim / 32);
    device_sparse_spmv<<<grid_dim, block_dim>>>(htrans,halpha,hbeta,hm,hn,hrowptr,hcolindex,hvalue,hx,hy);

	int deviceCount;
	hipGetDeviceCount(&deviceCount);

	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		int driver_version(0), runtime_version(0);
		hipDeviceProp_t deviceProp;
		hipGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor == 9999 && deviceProp.major == 9999)
				printf("\n");
		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
		hipDriverGetVersion(&driver_version);
		printf("hip驱动版本:                                       %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		hipRuntimeGetVersion(&runtime_version);
		printf("hip运行时版本:                                     %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("设备计算能力:                                      %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Total amount of Global Memory:                     %zu bytes\n", deviceProp.totalGlobalMem);
		printf("Number of SMs:                                     %d\n", deviceProp.multiProcessorCount);
		printf("Total amount of Constant Memory:                   %zu bytes\n", deviceProp.totalConstMem);
		printf("Total amount of Shared Memory per block:           %zu bytes\n", deviceProp.sharedMemPerBlock);
		printf("Total amount of Shared Memory per Multiprocessor:  %zu bytes\n", deviceProp.maxSharedMemoryPerMultiProcessor);
		printf("Total number of registers available per block:     %d\n", deviceProp.regsPerBlock);
		printf("Warp size:                                         %d\n", deviceProp.warpSize);
		printf("Maximum number of threads per SM:                  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("Maximum number of threads per block:               %d\n", deviceProp.maxThreadsPerBlock);
		printf("Maximum size of each dimension of a block:         %d x %d x %d\n", deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid:          %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
//		printf("Maximum memory pitch:                              %zu bytes\n", deviceProp.memPitch);
//		printf("Texture alignmemt:                                 %zu bytes\n", deviceProp.texturePitchAlignment);
		printf("Clock rate:                                        %.2f GHz\n", deviceProp.clockRate * 1e-6f);
		printf("Memory Clock rate:                                 %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("Memory Bus Width:                                  %d-bit\n", deviceProp.memoryBusWidth);
	}

}