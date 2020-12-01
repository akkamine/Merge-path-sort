#include <stdlib.h>
#include <math.h>

// Tableau de 12000 int max

#define N 512
#define NTPB 1024

__global__ void mergeSmall_k(int *a, int *b, int *m, int sizeA, int sizeB){
  int K[2];
  int P[2];
  int Q[2];

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ int sA [sizeA];
  __shared__ int sB [sizeB];

  sA[i] = a[i];
  sB[i] = b[i];

  __synchthreads();

  if(i<2*N){

    if(i>sizeA){
      K[0] = i - sizeA; K[1] = sizeA;
      P[0] = sizeA;     P[1] = i - sizeA;
    }
    else{
      K[0] = 0; K[1] = i;
      P[0] = i; P[1] = 0;
    }

    while(1){
      offset = abs(K[1]-P[1])/2;
      Q[0] = K[0] + offset;
      Q[1] = K[1] - offset;

      if(Q[1] >= 0 && Q[0] <= sizeB && (Q[1] = sizeA || Q[0] == 0 || sA[ Q[1] ] > sB[ Q[0]-1 ]){
        if(Q[0] == sizeB || Q[1] == 0 || sA[ Q[1]-1 ] <= sB[ Q[0] ]){
          if(Q[1] < sizeA && (Q[0] == sizeB || sA[ Q[1] ] <= sB[ Q[0] ])){
            M[i] = sA[ Q[1] ];
          }
          else{
            M[i] = sB[ Q[0] ];
          }
          break;
        }
        else{
          K[0] = Q[0] + 1;
          K[1] = Q[1] - 1;
        }
      }
      else{
        P[0] = Q[0] - 1;
        P[1] = Q[1] + 1;
      }
    }
  }
}

int main(void)
{
  int A[N], B[N], M[N];

  float time= 0.;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  for (int i = 0; i < N; i++){
    A[i] = i;
    B[i] = i+1;
  }

  int *A_gpu;
  int *B_gpu;
  int *M_gpu;

  cudaMalloc(&A_gpu, N* sizeof(int));
  cudaMalloc(&B_gpu, N* sizeof(int));
  cudaMalloc(&M_gpu, 2*N* sizeof(int));

  // run version with static shared memory
  cudaMemcpy(A_gpu, A, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, N*sizeof(int), cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  int nb_block = (N+NTPB-1)/NTPB;
  mergeSmall_k<<<nb_block,NTPB>>>(A, B, N, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("mergeSmall_k: temps écoulé = %f secs\n", time/1000);

  cudaMemcpy(M, M_gpu, 2*N*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
    printf("M[%d] = %d\n", i, M[i]);


  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(M_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
