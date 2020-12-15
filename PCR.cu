#include <stdio.h>
#include <time.h>
#include <math.h>

#define N 100
#define NTPB 1024


#define TPB 64
#define RAD 1 // radius of the stencil

__global__ void forwardPCR(float *a_in, float *b_in, float *c_in, float*d_in, float *a_out, float *b_out, float *c_out, float*d_out, int k, int size){
    
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i-k < 0){
        float k2 = c_in[i] / b_in[i+k];
        a_out[i] = a_in[i];
        b_out[i] = b_in[i] - a_in[i+k] * k2;
        c_out[i] = - c_in[i+k] * k2;
        d_out[i] = d_in[i] - d_in[i+k] * k2;
        return;
    }
    
    if(i+k >= size){
        float k1 = a_in[i] / b_in[i-k];
        a_out[i] = -a_in[i-k] * k1;
        b_out[i] = b_in[i] - c_in[i-k] * k1;
        c_out[i] = c_in[i];
        d_out[i] = d_in[i] - d_in[i-k] * k1;
        return;
    }
    
    float k1 = a_in[i] / b_in[i-k];
    float k2 = c_in[i] / b_in[i+k];
    
    a_out[i] = -a_in[i-k] * k1;
    b_out[i] = b_in[i] - c_in[i-k] * k1 - a_in[i+k] * k2;
    
    c_out[i] = - c_in[i+k] * k2;
    d_out[i] = d_in[i] - d_in[i-k] * k1 - d_in[i+k] * k2;
}

__global__ void solve2unknown(float *a, float *b, float *c, float*d, float *x, int k, int size){
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i > size/2)
        return;
    
    x[i+k] = ( d[i+k] - a[i+k]*d[i]/b[i] ) / ( b[i+k] - a[i+k]*c[i]/b[i] );
    x[i] = (d[i] - c[i]*x[i+k]) / b[i];
    
}
    
        
    

int main(){
    srand(time(NULL));
    
    float *a_cpu, *b_cpu, *c_cpu, *d_cpu, *x_cpu;
    float *a_gpu, *b_gpu, *c_gpu, *d_gpu, *x_gpu;
    float *a_out, *b_out, *c_out, *d_out, *x_out;
    
    // a 1ere sous diagonale | b main diag | c 1ere up diag
    // d right hand vector | x solution
  	a_cpu = (float*) malloc(N*sizeof(float));
    b_cpu = (float*) malloc(N*sizeof(float));
    c_cpu = (float*) malloc(N*sizeof(float));
    d_cpu = (float*) malloc(N*sizeof(float));
    x_cpu = (float*) malloc(N*sizeof(float));
    
    cudaMalloc(&a_gpu, N*sizeof(float));
    cudaMalloc(&b_gpu, N*sizeof(float));
    cudaMalloc(&c_gpu, N*sizeof(float));
    cudaMalloc(&d_gpu, N*sizeof(float));
    cudaMalloc(&x_gpu, N*sizeof(float));
    
    cudaMalloc(&a_out, N*sizeof(float));
    cudaMalloc(&b_out, N*sizeof(float));
    cudaMalloc(&c_out, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMalloc(&x_out, N*sizeof(float));
    
    int i, k;
    
    for(i=0; i<N; i++){
        a_cpu[i] = (float)rand()/(float)(RAND_MAX/10);
        b_cpu[i] = (float)rand()/(float)(RAND_MAX/10);
        c_cpu[i] = (float)rand()/(float)(RAND_MAX/10);
        d_cpu[i] = (float)rand()/(float)(RAND_MAX/10);
    }
    a_cpu[0] = 0;
    c_cpu[N-1] = 0;
    
    cudaMemcpy(a_gpu, a_cpu, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu, c_cpu, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu, d_cpu, N*sizeof(float), cudaMemcpyHostToDevice);

    k = 2;
    while(N/k > 1){
        forwardPCR<<<(N+TPB-1)/TPB, TPB>>>(a_gpu, b_gpu, c_gpu, d_gpu, a_out, b_out, c_out, d_out, k, N);
        cudaDeviceSynchronize();
        cudaMemcpy(a_gpu, a_out, N*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(b_gpu, b_out, N*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(c_gpu, c_out, N*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_gpu, d_out, N*sizeof(float), cudaMemcpyDeviceToDevice);
        k *= 2;
    }
    
    solve2unknown<<<(N+TPB-1)/TPB, TPB>>>(a_gpu, b_gpu, c_gpu, d_gpu, x_gpu, k, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(x_cpu, x_gpu, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    
    for(i=0; i<N; i++){
        printf("x_cpu[%d] = %f\n", i, x_cpu[i]);
    }
    
    free(a_cpu);
    free(b_cpu);
    free(c_cpu);
    free(d_cpu);
    free(x_cpu);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
    cudaFree(d_gpu);
    cudaFree(x_gpu);
    
    return 0;
}
