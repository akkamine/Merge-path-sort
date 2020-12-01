#include <stdio.h>

#define N 6144
#define NTPB 1024


__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[N];
  int t = threadIdx.x + blockIdx.x * blockDim.x;
  int tr = N-t-1;
  s[t] = d[t];
  __syncthreads();

  d[t] = s[tr];


__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x + blockIdx.x * blockDim.x;
  int tr = N-t-1;
  s[t] = d[t];
  __syncthreads();

  d[t] = s[tr];
}

int main(void)
{
  int a[N], r[N], d[N];

  float time= 0.;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  for (int i = 0; i < N; i++) {
    a[i] = i;
    r[i] = N-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, N* sizeof(int));

  // run version with static shared memory
  cudaMemcpy(d_d, a, N*sizeof(int), cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  int nb_block = (N+NTPB-1)/NTPB;
  staticReverse<<<nb_block,NTPB>>>(d_d, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("staticReverse: temps écoulé = %f secs\n", time/1000);

  cudaMemcpy(d, d_d, N*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++)
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

  // ruNdynamic shared memory version
  cudaMemcpy(d_d, a, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaEventRecord(start);

  dynamicReverse<<<nb_block,NTPB>>>(d_d, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("dynamicReverse: temps écoulé = %f secs\n", time/1000);

  cudaMemcpy(d, d_d, N* sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++)
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

  cudaFree(d_d);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
