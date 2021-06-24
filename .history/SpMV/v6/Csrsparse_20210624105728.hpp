#include<iostream>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <hip/hip_runtime.h>

enum sparse_operation {operation_none=0,operation_transpose=1} ;


__global__ void spmv_pcsr_kernel1(__SIZEOF_LONG_DOUBLE__ * d_val,__SIZEOF_LONG_DOUBLE__ * d_vector,int * d_cols,int d_nnz, __SIZEOF_LONG_DOUBLE__ * d_v)
{
    	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    	int icr = blockDim.x * gridDim.x;
    	while (tid < d_nnz){
		d_v[tid] = d_val[tid] * d_vector[d_cols[tid]];
        	tid += icr;
    	}
}

__global__ void spmv_pcsr_kernel2(T * d_v,int * d_ptr,int N,T * d_out)
{
    	int gid = blockIdx.x * blockDim.x + threadIdx.x;
    	int tid = threadIdx.x;
    
    	__shared__ volatile int ptr_s[threadsPerBlock + 1];
    	__shared__ volatile T v_s[sizeSharedMemory];
 
   	// Load ptr into the shared memory ptr_s
    	ptr_s[tid] = d_ptr[gid];

	// Assign thread 0 of every block to store the pointer for the last row handled by the block into the last shared memory location
    	if (tid == 0) { 
    		if (gid + threadsPerBlock > N) {
	    		ptr_s[threadsPerBlock] = d_ptr[N];}
		else {
    	    		ptr_s[threadsPerBlock] = d_ptr[gid + threadsPerBlock];}
    	}
    	__syncthreads();

    	int temp = (ptr_s[threadsPerBlock] - ptr_s[0])/threadsPerBlock + 1;
    	int nlen = min(temp * threadsPerBlock,sizeSharedMemory);
    	T sum = 0;
    	int maxlen = ptr_s[threadsPerBlock];     
    	for (int i = ptr_s[0]; i < maxlen; i += nlen){
    		int index = i + tid;
    		__syncthreads();
    		// Load d_v into the shared memory v_s
    		for (int j = 0; j < nlen/threadsPerBlock;j++){
	    		if (index < maxlen) {
	        		v_s[tid + j * threadsPerBlock] = d_v[index];
	        		index += threadsPerBlock;
            		}
    		}
   	 	__syncthreads();

    		// Sum up the elements for a row
		if (!(ptr_s[tid+1] <= i || ptr_s[tid] > i + nlen - 1)) {
	   		int row_s = max(ptr_s[tid] - i, 0);
	    		int row_e = min(ptr_s[tid+1] -i, nlen);
	    		for (int j = row_s;j < row_e;j++){
				sum += v_s[j];
	    		}
		}	
    	}	
	// Write result
    	d_out[gid] = sum;
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
	int block_dim = 1024;
	int *d_temp;
	hipMalloc((void **)&d_temp, sizeof(int));

	spmv_pcsr_kernel1<T><<<ceil(NonZ/(float)block_dim),block_dim>>>(d_val,d_vector,d_cols,NonZ,d_temp);
    spmv_pcsr_kernel2<T><<<ceil(hm/(float)64),64>>>(d_v,d_ptr,hm,d_out);
}