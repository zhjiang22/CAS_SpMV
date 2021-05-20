#include<iostream>
#include<cstdlib>
#include<ctime>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <hip/hip_runtime.h>
#include <math.h>
#include"./Csrsparse.hpp"
#include "./common_function.hpp"
#include <sys/time.h>
#include <unistd.h>

using namespace std;

#define gpu

// 定义计时器
struct my_timer
{
    struct timeval start_time, end_time;
    double time_use; // us
    void start(){
		gettimeofday(&start_time, NULL);
    }
    void stop(){
		gettimeofday(&end_time, NULL);
		time_use = (end_time.tv_sec-start_time.tv_sec)*1.0e6 + end_time.tv_usec-start_time.tv_usec;
    }	
};

int main(int argc,char** argv){
    srand(1);
    if (argc!=4)
    {
        cout<<"请输入运行参数,第一个参数为矩阵维度m,第二个参数为矩阵维度n,第三个参数为稠密度(稀疏度+稠密度=1)"<<endl;
        return 1;
    }

    //n为矩阵维度N*N
    m=atoi(argv[1]);
    n=atoi(argv[2]);
    //s为稠密度,s越大0越少
    s=atof(argv[3]);
    //s = rand_double(0,1);
    // alpha = rand_integer(1,10);
    // beta=rand_integer(1,10);
    //beta=0;
    cout<<"稀疏度:"<<(1-s)<<" alpha:"<<alpha<<" beta:"<<beta<<endl;

    hipSetDevice(0);
    
	create_host_data();
    create_deivce_data();


    //--------------------------------------------------------------------------
    
    //warm up 硬件预热
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse spmv
        HIP_CHECK( hipMemcpy(dY, temphY, A_num_rows * sizeof(double),
                           hipMemcpyHostToDevice) )
        sparse_spmv(operation,alpha,beta,A_num_rows,A_num_cols,dA_csrOffsets,dA_columns,dA_values,dX,dY);
    }
    hipDeviceSynchronize();
	double sum_costs = 0.0;
    my_timer timer1;
	//计时开始
	timer1.start();
    // execute device SpMV 
    //for (int i=0;i<1;i++)
    //{       
            sparse_spmv(operation,alpha,beta,A_num_rows,A_num_cols,dA_csrOffsets,dA_columns,dA_values,dX,dY);
            hipDeviceSynchronize();	
    //}
	//计时结束
	timer1.stop();
    // device result check
    HIP_CHECK( hipMemcpy(dY, temphY, A_num_rows * sizeof(double),
                           hipMemcpyHostToDevice) )
    sparse_spmv(operation,alpha,beta,A_num_rows,A_num_cols,dA_csrOffsets,dA_columns,dA_values,dX,dY);
    HIP_CHECK( hipMemcpy(hY, dY, A_num_rows * sizeof(double),
                           hipMemcpyDeviceToHost) );
    #ifdef gpu
    //设备端端验证
    HIP_CHECK( hipMemcpy(dY, temphY, A_num_rows * sizeof(double),
                           hipMemcpyHostToDevice) )
    my_timer timer2;
    timer2.start();
    rocsparse()  ;                     
    timer2.stop();
    HIP_CHECK( hipMemcpy(hhY, dY, A_num_rows * sizeof(double),
                          hipMemcpyDeviceToHost) );
    #else
    //主机端验证
    spmv(alpha,beta,value,rowptr,colindex,m,n,a,hX,hhY);
    //print_vector(n,hhY);
    #endif 
    verify(hY,hhY,m);
	cout<<"elapsed time:"<<timer1.time_use<<"(us)"<<endl;
	cout<<"elapsed time:"<<timer2.time_use<<"(us)"<<endl;
	// printf("hy as flows\n");
    // print_vector(n,hY);
    return 0;
}
