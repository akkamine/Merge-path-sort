#include <stdlib.h>
#include <math.h>
#include <stdio.h>


#define N_THREADS 1024
#define N_BLOCKS 64

#define X 0
#define Y 1

/*********************************************/
/************* PATH BIG K ********************/
/*********************************************/

__global__ void pathBig_k(int *a, int *b, int *Aindex, int *Bindex, int sizeA, int sizeB){
    // cette fonction permet de construire Aindex, Bindex
    // Aindex et Bindex sont stockés en mémoire globale
    // ils permettent de stocker les points de ruptures entre threads
    
    // cette fonction est exécutée par un seul thread par block
    
    // i prend la valeur de cette diagonale
    const int i = (sizeA+sizeB)/N_BLOCKS * blockIdx.x;
    
	int K[2];
	int P[2];
	int Q[2];
    
    if(i>sizeA){
        K[X] = i - sizeA;
        K[Y] = sizeA;
        P[X] = sizeA;
        P[Y] = i - sizeA;
    }
    else{
        K[X] = 0; K[Y] = i;
        P[X] = i; P[Y] = 0;
    }

    while(1){

        int offset = (K[Y]-P[Y])/2;
        Q[X] = K[X] + offset;
        Q[Y] = K[Y] - offset;

        if(Q[Y] >= 0 && Q[X] <= sizeB && (Q[Y] == sizeA || Q[X] == 0 || a[ Q[Y] ] > b[ Q[X]-1 ]) ){
            if(Q[X] == sizeB || Q[Y] == 0 || a[ Q[Y]-1 ] <= b[ Q[X] ]){
                Aindex[blockIdx.x] = Q[Y];
                Bindex[blockIdx.x] = Q[X];
                break;
            }
            else{
                K[X] = Q[X] + 1;
                K[Y] = Q[Y] - 1;
            }
        }
        else{
            P[X] = Q[X] - 1;
            P[Y] = Q[Y] + 1;
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

	const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int local_i = threadIdx.x;

    // Taille des sous-tableaux de A et B stockés en mémoire shared du block
    int sizeSA = Aindex[blockIdx.x + 1] - Aindex[blockIdx.x];
    int sizeSB = Bindex[blockIdx.x + 1] - Bindex[blockIdx.x];
    
	if(local_i>=(sizeSA + sizeSB))
        return;

	__shared__ int sharedMem[1024]; //de taille sizeSA + sizeSB
	
	int *sA = sharedMem;
    int *sB = sizeSA*sizeof(int) + sharedMem;
    
    if(blockIdx.x == N_BLOCKS - 1){
        sA[local_i % sizeSA] = a[i%sizeA];
        sB[local_i % sizeSB] = b[i % sizeB];
    }
    
    else{
        if(Aindex[blockIdx.x] <= i <= Aindex[blockIdx.x + 1])
            sA[local_i] = a[i];
        if(Bindex[blockIdx.x] <= i <= Bindex[blockIdx.x + 1])
            sB[local_i] = b[i];
    }
    
    __syncthreads();
    
    if(local_i >= sizeSA + sizeSB)
        return;
    
    if(local_i>sizeSA){
        K[X] = local_i - sizeSA;
        K[Y] = sizeSA;
        P[X] = sizeSA;
        P[Y] = local_i - sizeSA;
    }
    else{
        K[X] = 0;       K[Y] = local_i;
        P[X] = local_i; P[Y] = 0;
    }

    while(1){

        int offset = (K[1]-P[1])/2;
        Q[X] = K[X] + offset;
        Q[Y] = K[Y] - offset;

        if(Q[Y] >= 0 && Q[X] <= sizeSB && (Q[Y] == sizeSA || Q[X] == 0 || sA[ Q[Y] ] > sB[ Q[X]-1 ]) ){
            if(Q[X] == sizeSB || Q[Y] == 0 || sA[ Q[Y]-1 ] <= sB[ Q[X] ]){
					if(Q[Y] < sizeSA && (Q[X] == sizeSB || sA[ Q[Y] ] <= sB[ Q[X] ])){
                        m[i] = sA[Q[Y]] ;
		  			}
		  			else{
						m[i] = sB[Q[X]];
		  			}
		  			break;
            }
            else{
                K[X] = Q[X] + 1;
                K[Y] = Q[Y] - 1;
            }
        }
        else{
            P[X] = Q[X] - 1;
            P[Y] = Q[Y] + 1;
        }
    }
}

int main(void){
    int sizeA = 65536;
    int sizeB = 65536;
    
	int *A;
	int *B;
	int *M;
	//int nb_blocks = (N+NTPB-1)/NTPB; // A vérifier
    int nb_blocks = N_BLOCKS;
    
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
    
    // recherche du découpage 
    pathBig_k<<<N_BLOCKS, 1>>>(A_gpu, B_gpu, Aindex, Bindex, sizeA, sizeB);
    
    mergeBig_k<<<N_BLOCKS, N_THREADS>>>(A_gpu, B_gpu, M_gpu, Aindex, Bindex, sizeA, sizeB);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("mergeBig_k: temps écoulé = %f secs\n", time/1000);


	cudaMemcpy(M, M_gpu, (sizeA+sizeB)*sizeof(int), cudaMemcpyDeviceToHost);
    
	for (int i = 0; i < (sizeA+sizeB); i++)
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
