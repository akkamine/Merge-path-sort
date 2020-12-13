#include <stdio.h>
#include <time.h>
#include <math.h>

#define N 100
#define NTPB 1024
#define RAD 1

#define TPB 64
#define RAD 1 // radius of the stencil

__global__ void forwardCR(float *a, float *b, float *c, float*d, int k){
    
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(((i+1) % k) != 0)
        return;
    
    float k1 = a[i] / b[i-k];
    float k2 = c[i] / b[i+k];
    
    a[i] = -a[i-k] * k1;
    b[i] = b[i] - c[i-k] * k1 - a[i+k] * k2;
    
    c[i] = - c[i+k] * k2;
    d[i] = d[i] - d[i-k] * k1 - d[i+k] * k2;
}
    
__global__ void backwardCR(float *a, float *b, float *c, float*d, float *x, int k){
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(((i+1) % k) != 0)
        return;
    
    x[i] = ( d[i] - a[i] * x[i-k] - c[i] * x[i+1] ) /b[i];
}

__global__ void solve2unknown(float *a, float *b, float *c, float*d, float *x, int k){
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(((i+1) % k) != 0)
        return;
    
    x[2*k] = ( d[2*k] - a[2*k]*d[k]/b[k] ) / ( b[2*k] - a[2*k]*c[k]/b[k] );
    x[k] = (d[k] - c[k]*x[2*k]) / b[k];
    
}
    
        
    

int main(){
    srand(time(NULL));
    
    float *a_cpu, *b_cpu, *c_cpu, *d_cpu, *x_cpu;
    float *a_gpu, *b_gpu, *c_gpu, *d_gpu, *x_gpu;
    
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
        forwardCR<<<(N+TPB-1)/TPB, TPB>>>(a_gpu, b_gpu, c_gpu, d_gpu, k);
        cudaDeviceSynchronize();
        k *= 2;
    }
    
    solve2unknown<<<(N+TPB-1)/TPB, TPB>>>(a_gpu, b_gpu, c_gpu, d_gpu, x_gpu, k);
    cudaDeviceSynchronize();
    k /= 2;
    
    
    while(N/k < N){
        backwardCR<<<(N+TPB-1)/TPB, TPB>>>(a_gpu, b_gpu, c_gpu, d_gpu, x_gpu, k);
        cudaDeviceSynchronize();
        k /= 2;
    }
    
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


/*
__global__ void ddKernel(float *d_out, const float *d_in, int size, float h){
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i>= size) return;
    
    const int s_idx = threadIdx.x + RAD;
    extern __shared__ float s_in[];
    
    // REgular cells
    s_in[s_idx] = d_in[i];
    //Halo cells
    if(threadIdx.x < RAD){
        s_in[s_idx - RAD] = d_in[i-RAD];
        s_in[s_idx + blockDim.x] = d_in[i+blockDim.x];
    }
    __syncthreads();
    d_out[i] = (s_in[s_idx-1] - 2.f*s_in[s_idx] + s_in[s_idx+1])/(h*h);
}


void ddParallel(float *out, const float *in, int n, float h){
    float *d_in = 0, *d_out = 0;
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMalloc(&d_out, n*sizeof(float));
    cudaMemcpy(d_in, in, n*sizeof(float), cudaMemcpyHostToDevice);
    
    // Set shared memory size in bytes
    const size_t smemSize = (TPB + 2*RAD) * sizeof(float);
    ddKernel<<<(n+TPB-1)/TPB, TPB, smemSize>>>(d_out, d_in, n, h);
    
    cudaMemcpy(out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

int main(){
    float *in_cpu, *out_cpu;
	in_cpu = (float*) malloc(N*sizeof(float));
    out_cpu = (float*) malloc(N*sizeof(float));
    int i;
	for(i=0; i<N; i++){
		in_cpu[i] = i;
		out_cpu[i] = 4;
	}
	
	ddParallel(out_cpu, in_cpu, N, 0.33);
    cudaDeviceSynchronize();
    for(i=0; i<100; i++)
        printf("out_cpu[%d] = %f\n", i, out_cpu[i]);
    return 0;
}
*/
