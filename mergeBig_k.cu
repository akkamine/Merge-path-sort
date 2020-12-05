#include <stdlib.h>
#include <math.h>
#include <stdio.h>


#define N 512
#define NTPB 1024

/*********************************************/
/************* PATH BIG K ********************/
/*********************************************/

__global__ void pathBig_k(int *a, int *b, int *Aindex, int *Bindex, int sizeA, int sizeB){
    // indexA et indexB sont stockés en mémoire globale
    // ils permettent de stocker les points de ruptures entre threads
	int K[2];
	int P[2];
	int Q[2];

	int i = threadIdx.x + blockIdx.x * blockDim.x;    
    
    // Déterminer condition limite. A priori c'est ça
	if(i>=sizeA+sizeB)
        return;
    
    // Exécuté par un seul thread par block
    if(i%NTPB != 0)
        return;
    
    if(i>sizeA){
        K[0] = i - sizeA;
        K[1] = sizeA;
        P[0] = sizeA;
        P[1] = i - sizeA;
    }
    else{
        K[0] = 0; K[1] = i;
        P[0] = i; P[1] = 0;
    }

    while(1){

        int offset = (K[1]-P[1])/2;
        Q[0] = K[0] + offset;
        Q[1] = K[1] - offset;

        if(Q[1] >= 0 && Q[0] <= sizeB && (Q[1] == sizeA || Q[0] == 0 || a[ Q[1] ] > a[ Q[0]-1 ]) ){
            if(Q[0] == sizeB || Q[1] == 0 || a[ Q[1]-1 ] <= b[ Q[0] ]){
                Aindex[blockIdx.x] = Q[0];
                Bindex[blockIdx.x] = Q[1];
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


/**********************************************/
/************* MERGE BIG K ********************/
/**********************************************/

__global__ void mergeBig_k(int *a, int *b, int *m, int *Aindex, int *Bindex, int sizeA, int sizeB){
	int K[2];
	int P[2];
	int Q[2];

	int i = threadIdx.x;// + blockIdx.x * blockDim.x;

	if(i>=(sizeA + sizeB))
        return;

    // Taille des sous-tableaux de A et B stockés en mémoire shared du block
    int sizeSA = Aindex[blockIdx.x + 1] - Aindex[blockIdx.x];
    int sizeSB = Bindex[blockIdx.x + 1] - Bindex[blockIdx.x];
    
	extern __shared__ int sharedMem[]; //de taille sizeSA + sizeSB
	
	int *sA = sharedMem;
    int *sB = sizeSA*sizeof(int) + sharedMem;
    
    if(blockIdx.x == ((sizeA+sizeB)/NTPB)-1 && i < sizeA)
        sA[i%sizeSA] = a[i%sizeA];
    else
        if(Aindex[blockIdx.x] <= i <= Aindex[blockIdx.x + 1])
            sA[i%sizeSA] = a[i%sizeA];
    
    if(blockIdx.x == ((sizeA+sizeB)/NTPB)-1 && i < sizeA)
        sA[i%sizeSA] = a[i%sizeA];
    else
        if(Bindex[blockIdx.x] <= i <= Bindex[blockIdx.x + 1])
            sB[i%sizeSB] = b[i%sizeB];
    
    __syncthreads();
    
    if(i>sizeA){
        K[0] = i - sizeSA;
        K[1] = sizeSA;
        P[0] = sizeSA;
        P[1] = i - sizeSA;
    }
    else{
        K[0] = 0; K[1] = i;
        P[0] = i; P[1] = 0;
    }

    while(1){

        int offset = (K[1]-P[1])/2;
        Q[0] = K[0] + offset;
        Q[1] = K[1] - offset;

        if(Q[1] >= 0 && Q[0] <= sizeSB && (Q[1] == sizeSA || Q[0] == 0 || sA[ Q[1] ] > sB[ Q[0]-1 ]) ){
            if(Q[0] == sizeSB || Q[1] == 0 || sA[ Q[1]-1 ] <= sB[ Q[0] ]){
					if(Q[1] < sizeSA && (Q[0] == sizeSB || sA[ Q[1] ] <= sB[ Q[0] ])){
                        m[i] = sA[Q[1]] ;
		  			}
		  			else{
						m[i] = sB[Q[0]];
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

int main(void){
    int sizeA = N;
    int sizeB = N;
    
	int *A;
	int *B;
	int *M;
	int nb_blocks = (N+NTPB-1)/NTPB; // A vérifier

	float time= 0.;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	A = (int*) malloc(sizeof(int)*sizeA);
	B = (int*) malloc(sizeof(int)*sizeB);
	M = (int*) calloc(sizeof(int),sizeA+sizeB);

	for (int i = 0; i < sizeA; i++)
		A[i] = i*2+1;
    
    for (int i = 0; i < sizeB; i++)
		B[i] = i*2;

	int *A_gpu;
	int *B_gpu;
	int *M_gpu;
    
    int *Aindex, *Bindex;

	cudaMalloc(&A_gpu, sizeA * sizeof(int));
	cudaMalloc(&B_gpu, sizeB * sizeof(int));
	cudaMalloc(&M_gpu, (sizeA+sizeB) * sizeof(int));

    cudaMalloc(&Aindex, nb_blocks * sizeof(int));
	cudaMalloc(&Bindex, nb_blocks * sizeof(int));

	cudaMemcpy(A_gpu, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
    
    pathBig_k<<<1,NTPB>>>(A_gpu, B_gpu, Aindex, Bindex, sizeA, sizeB);
    
    mergeBig_k<<<1,NTPB>>>(A_gpu, B_gpu, M_gpu, Aindex, Bindex, sizeA, sizeB);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("mergeSmall_k: temps écoulé = %f secs\n", time/1000);


	cudaMemcpy(M, M_gpu, 2*N*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2*N; i++)
        printf("M[%d] = %d\n", i, M[i]);


	free(A);
	free(B);
	free(M);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(M_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
