#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define X 0
#define Y 1
#define SIZEA 1123
#define SIZEB 2223

#define N_BLOCKS 64
#define N_THREADS 2

__global__ void pathBig_k(const int *A, const int *B, int *Aindex, int *Bindex, const int sizeA, const int sizeB, const int morceaux){

    if(blockIdx.x == 0){
        Aindex[0] = 0;
        Bindex[0] = 0;
        Aindex[morceaux] = sizeA;
        Bindex[morceaux] = sizeB;
        return;
    }
    
	int i = (sizeA + sizeB)/morceaux * blockIdx.x;
	int K[2];
	int P[2];
    int Q[2];
    int offset;

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
		offset = (abs(K[Y] - P[Y]))/2;
		Q[X] = K[X] + offset;
		Q[Y] = K[Y] - offset;

		if (Q[Y] >= 0 && Q[X] <= sizeB && (Q[Y] == sizeA || Q[X] == 0 || A[Q[Y]] > B[Q[X]-1])) {
			if (Q[X] == sizeB || Q[Y] == 0 || A[Q[Y]-1] <= B[Q[X]]) {
				Aindex[blockIdx.x] = Q[Y];
				Bindex[blockIdx.x] = Q[X];
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

__global__ void mergeBig_k(int *A, int *B, int *M, int *Aindex, int *Bindex){
    
    int i = threadIdx.x;

	// Mémoire shared sur laquelle on va travailler
	__shared__ int A_shared[N_THREADS];
	__shared__ int B_shared[N_THREADS];

    // Biais de tour correspondant à un thread
    int biaisAi; // Décalage induit ou non par le thread (0 ou 1)
    int biaisBi;
    
    // Biais totaux
    __shared__ int biaisA;
    __shared__ int biaisB;
    
	int startABlock = Aindex[blockIdx.x];
	int endABlock = Aindex[blockIdx.x+1];
	int startBBlock = Bindex[blockIdx.x];
	int endBBlock = Bindex[blockIdx.x+1];

    // Taille des partitions de A et B
    int sABlock = endABlock - startABlock;
    int sBBlock = endBBlock - startBBlock;

    // Nombre de fenêtres glissantes
	int nb_windows = (blockDim.x - 1 + sABlock + sBBlock) / blockDim.x;
    biaisAi = 0;
    biaisBi = 0;
    
    biaisA = 0;
    biaisB = 0;
    // Merge fenêtre par fenêtre
    for(int k=0; k < nb_windows; k++){
        
        // Somme des biais de A et de B
        biaisA += __syncthreads_count(biaisAi);
        biaisB += __syncthreads_count(biaisBi);
        
        // Réinitialisation des biais de thread
        biaisAi = 0;
        biaisBi = 0;
    
        // Copie en mémoire shared
		if (startABlock + biaisA + i < endABlock)
			A_shared[i] = A[startABlock + biaisA + i];

		if (startBBlock + biaisB + i < endBBlock)
			B_shared[i] = B[startBBlock + biaisB + i];	

		// Synchronisation de la mémoire shared
		__syncthreads();
        
        // Taille des sous tableaux en mémoire shared
		int sizeAshared = min(blockDim.x, max(0, sABlock - biaisA));
		int sizeBshared = min(blockDim.x, max(0, sBBlock - biaisB));
        
		// Recherche dichotomique
		if (i < (sizeAshared + sizeBshared)){
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
						if (Q[Y] < sizeAshared && (Q[X] == sizeBshared || A_shared[Q[Y]] <= B_shared[Q[X]]) ) {
							M[i + startABlock + startBBlock + k * blockDim.x] = A_shared[Q[Y]];
                            biaisAi += 1;
						}
						else {
							M[i + startABlock + startBBlock + k * blockDim.x] = B_shared[Q[X]];
                            biaisBi += 1;
						}
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
	}
}



void sortBig_k(int *a_gpu, int sizeA){
    
    // Cas limites
    if(sizeA == 1)
        return;
    
    // On découpe le tableau a en 2
    int sizeA0 = (int) sizeA/2;
    int sizeA1 = sizeA - sizeA0;
    
    // On appelle récursivement sortBig_k sur les deux sous tableaux de a
    // En sortie on récupère deux tableaux triés donc on peut lancer pathBig et mergeBig
    sortBig_k(a_gpu, sizeA0);
    sortBig_k(a_gpu + sizeA0, sizeA1);
    
    // Déclaration 
	int *a0_gpu, *a1_gpu, *m_gpu, *A0index, *A1index;

	// Allocation de la mémoire globale du GPU
	cudaMalloc( (void**) &a0_gpu, sizeA0 * sizeof(int) );
	cudaMalloc( (void**) &a1_gpu, sizeA1 * sizeof(int) );
	cudaMalloc( (void**) &m_gpu, (sizeA) * sizeof(int) );
	cudaMalloc( (void**) &A0index, (N_BLOCKS+1) * sizeof(int) );
	cudaMalloc( (void**) &A1index, (N_BLOCKS+1) * sizeof(int) );

	cudaMemcpy(a0_gpu, a_gpu,          sizeA0 * sizeof(int), cudaMemcpyDeviceToDevice );
	cudaMemcpy(a1_gpu, a_gpu + sizeA0, sizeA1 * sizeof(int), cudaMemcpyDeviceToDevice );
    
	pathBig_k<<<N_BLOCKS, 1>>>(a0_gpu, a1_gpu, A0index, A1index, sizeA0, sizeA1, N_BLOCKS);
	mergeBig_k<<<N_BLOCKS, N_THREADS>>>(a0_gpu, a1_gpu, m_gpu, A0index, A1index);
    
    cudaMemcpy(a_gpu, m_gpu, sizeA * sizeof(int), cudaMemcpyDeviceToDevice );
    
    cudaFree(a0_gpu);
	cudaFree(a1_gpu);
	cudaFree(m_gpu);
    cudaFree(A0index);
    cudaFree(A1index);
    
}


int main(){
    srand(time(NULL));
	// Allocation de la mémoire, remplissage du tableau
	int *A = (int*) malloc(sizeof(int) * SIZEA);
    int *A_gpu;
	for (int i = 0; i < SIZEA; i++){
		A[i] = rand();
	}
	
	cudaMalloc( (void**) &A_gpu, SIZEA * sizeof(int) );
	cudaMemcpy(A_gpu, A, SIZEA * sizeof(int), cudaMemcpyHostToDevice);

	sortBig_k(A_gpu,SIZEA);
    
    cudaMemcpy(A, A_gpu, SIZEA * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < SIZEA; i ++){
		printf("A[%d] = %d\n", i, A[i]);
	}

	// Liberation de la mémoire
	free(A);

	return 0;
}
