#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define X 0
#define Y 1
#define SIZEA 1024
#define SIZEB 2048

#define N_BLOCKS 64
#define N_THREADS 2

__global__ void mergeBig_k(int *A, int *B, int sizeA, int sizeB, int *M, int *A_idx, int *B_idx){

	// Mémoire shared sur laquelle nous allons travaillé
	__shared__ int A_shared[1024];
	__shared__ int B_shared[1024];

	__shared__ int biaisA;
	__shared__ int biaisB;

	// (endA-startA) : taille de A dans la partition
	// (endB-startB) : taille de B dans la partition
	int startA, endA;
	int startB, endB;
	
	// On récupére les index du début et de la fin de A et B par rapport au tableau global
	if (blockIdx.x == 0){
		startA = 0;
		endA = A_idx[blockIdx.x];
		startB = 0;
		endB = B_idx[blockIdx.x];
	}
	else if (blockIdx.x == N_BLOCKS-1){
		startA = A_idx[blockIdx.x-1];
		endA = sizeA;
		startB = B_idx[blockIdx.x-1];
		endB = sizeB;
	}
	else{
		startA = A_idx[blockIdx.x-1];
		endA = A_idx[blockIdx.x];
		startB = B_idx[blockIdx.x-1];
		endB = B_idx[blockIdx.x];
	}

	// Notations de l'article
	// Il y a N élements à fusioner
	// N = SIZEA + SIZEB 
	// Chaque partition contient N/p éléments, chaque bloc traite une partition
	// N / p = (endB-startB) + (endA-startA) = (SIZEA+SIZEB) / N_BLOCKS
	// Si Z est le nombre de threads
	// On va fusioner Z éléments à la fois
	// Donc on a besoin de le faire (N / p) / Z fois
	// On va faire bouger la fenetre glissante (N / p) / Z fois
	int iter_max = (blockDim.x - 1 + (endB-startB) + (endA-startA)) / blockDim.x;
	int iter = 0;

	biaisA = 0;
	biaisB = 0;
	do{
		// Pour synchroniser les biais
		__syncthreads();

		// Chargement des valeurs dans la mémoire shared
		if (startA + biaisA + threadIdx.x < endA){
			A_shared[threadIdx.x] = A[startA + biaisA + threadIdx.x];
		}

		if (startB + biaisB + threadIdx.x < endB){
			B_shared[threadIdx.x] = B[startB + biaisB + threadIdx.x];	
		}

		// Pour synchroniser la mémoire shared
		__syncthreads();

		// Récuperer la taille de la fenetre glissante
		// En général c'est le nombre de threads (blockDim.x), i.e On est dans un carré Z * Z normalement
		// Mais la taille peut être inférieure si il y a moins de blockDim.x éléments à charger
		int sizeAshared = endA-startA - biaisA;
		int sizeBshared = endB-startB - biaisB;
		if (sizeAshared < 0)
			sizeAshared = 0;
		if (sizeAshared > blockDim.x && sizeAshared != 0)
			sizeAshared = blockDim.x;
		if (sizeBshared < 0)
			sizeBshared = 0;
		if (sizeBshared > blockDim.x && sizeBshared != 0)
			sizeBshared = blockDim.x;

		// Binary search
		int i = threadIdx.x;

		if (i < sizeAshared + sizeBshared){
			int K[2];
			int P[2];

			if (i > sizeAshared) {
				K[X] = i - sizeAshared;
				K[Y] = sizeAshared;
				P[X] = sizeAshared;
				P[Y] = i - sizeAshared;
			}
			else {
				K[X] = 0;
				K[Y] = i;
				P[X] = i;
				P[Y] = 0;
			}

			while (1) {
				int offset = (abs(K[Y] - P[Y]))/2;
				int Q[2] = {K[X] + offset, K[Y] - offset};

				if (Q[Y] >= 0 && Q[X] <= sizeBshared && (Q[Y] == sizeAshared || Q[X] == 0 || A_shared[Q[Y]] > B_shared[Q[X]-1])) {
					if (Q[X] == sizeBshared || Q[Y] == 0 || A_shared[Q[Y]-1] <= B_shared[Q[X]]) {
						int idx = startA + startB + i + iter * blockDim.x;
						if (Q[Y] < sizeAshared && (Q[X] == sizeBshared || A_shared[Q[Y]] <= B_shared[Q[X]]) ) {
							M[idx] = A_shared[Q[Y]];
							atomicAdd(&biaisA, 1);	// Biais à incrementer 
						}
						else {
							M[idx] = B_shared[Q[X]];
							atomicAdd(&biaisB, 1); // Biais à incrementer
						}
						//printf("blockIdx.x = %d threadIdx.x = %d idx = %d m = %d biaisA = %d\n", blockIdx.x, threadIdx.x, idx, M[idx], biaisA);
						break ;
					}
					else {
						K[X] = Q[X] + 1;
						K[Y] = Q[Y] - 1;
					}
				}
				else {
					P[X] = Q[X] - 1;
					P[Y] = Q[Y] + 1 ;
				}
			}
		}
		iter = iter + 1;
	} while(iter < iter_max);
}

__global__ void pathBig_k(int *A, int *B, int sizeA, int sizeB, int *M, int *A_idx, int *B_idx){

	// Dans ce kernel, on va simplement chercher N_BLOCKS diagonales
	// de telle sorte que chaque bloc traitera N / N_BLOCKS elements dans le second kernel
	int i = (sizeA + sizeB)/N_BLOCKS * (blockIdx.x + 1);
	if (blockIdx.x == N_BLOCKS-1){
		return;
	}

	// Binary search
	int K[2];
	int P[2];

	if (i > sizeA) {
		K[X] = i - sizeA;
		K[Y] = sizeA;
		P[X] = sizeA;
		P[Y] = i - sizeA;
	}
	else {
		K[X] = 0;
		K[Y] = i;
		P[X] = i;
		P[Y] = 0;
	}

	while (1) {

		int offset = (abs(K[Y] - P[Y]))/2;
		int Q[2] = {K[X] + offset, K[Y] - offset};

		if (Q[Y] >= 0 && Q[X] <= sizeB && (Q[Y] == sizeA || Q[X] == 0 || A[Q[Y]] > B[Q[X]-1])) {
			if (Q[X] == sizeB || Q[Y] == 0 || A[Q[Y]-1] <= B[Q[X]]) {
				if (Q[Y] < sizeA && (Q[X] == sizeB || A[Q[Y]] <= B[Q[X]]) ) {
					M[i] = A[Q[Y]];
				}
				else {
					M[i] = B[Q[X]];
				}
				A_idx[blockIdx.x] = Q[Y];
				B_idx[blockIdx.x] = Q[X];
				// printf("blockIdx.x = %d | Aidx[%d] = %d | Bidx[%d] = %d \n", blockIdx.x, blockIdx.x, Q[Y], blockIdx.x, Q[X]);
				break ;
			}
			else {
				K[X] = Q[X] + 1;
				K[Y] = Q[Y] - 1;
			}
		}
		else {
			P[X] = Q[X] - 1;
			P[Y] = Q[Y] + 1;
		}
	}
}


void sortBig_k(int *a, int sizeA){
    
    if(sizeA == 1)
        return;
    if(sizeA == 2){
        if(a[0] > a[1]){
            int temp = a[1];
            a[1] = a[0];
            a[0] = temp;
        }
        return;
    }
    
    int sizeA0 = (int) sizeA/2;
    int sizeA1 = sizeA - sizeA0;
    
    sortBig_k(a, sizeA0);
    sortBig_k(a + sizeA0, sizeA1);
    
	int *a0Device, *a1Device, *mDevice, *A0_idxDevice, *A1_idxDevice;

	// Allocation de la mémoire globale du GPU
	cudaMalloc( (void**) &a0Device, sizeA0 * sizeof(int) );
	cudaMalloc( (void**) &a1Device, sizeA1 * sizeof(int) );
	cudaMalloc( (void**) &mDevice, (sizeA) * sizeof(int) );
	cudaMalloc( (void**) &A0_idxDevice, N_BLOCKS * sizeof(int) );
	cudaMalloc( (void**) &A1_idxDevice, N_BLOCKS * sizeof(int) );

	cudaMemcpy(a0Device, a,          sizeA0 * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy(a1Device, a + sizeA0, sizeA1 * sizeof(int), cudaMemcpyHostToDevice );
    
	pathBig_k<<<N_BLOCKS, 1>>>(a0Device, a1Device, sizeA0, sizeA1, mDevice, A0_idxDevice, A1_idxDevice);
	mergeBig_k<<<N_BLOCKS, N_THREADS>>>(a0Device, a1Device, sizeA0, sizeA1, mDevice, A0_idxDevice, A1_idxDevice);
    
    cudaMemcpy(a, mDevice, sizeA * sizeof(int), cudaMemcpyDeviceToHost );
    
    cudaFree(a0Device);
	cudaFree(a1Device);
	cudaFree(mDevice);
    cudaFree(A0_idxDevice);
    cudaFree(A1_idxDevice);
    
}

int main(){
    srand(time(NULL));
	// Allocation de la mémoire, remplissage du tableau
	int *A = (int*) malloc(sizeof(int) * SIZEA);
	for (int i = 0; i < SIZEA; i++){
		A[i] = rand();
	}

	
	sortBig_k(A,SIZEA);

	for (int i = 0; i < SIZEA; i ++){
		printf("A[%d] = %d\n", i, A[i]);
	}

	// Liberation de la mémoire
	free(A);

	return 0;
}
