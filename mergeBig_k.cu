#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define X 0
#define Y 1
#define SIZEA 2024
#define SIZEB 2048

#define N_BLOCKS 64
#define N_THREADS 32

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


int main(){
    int i;
    
    // Allocation et replissage des tableaux d'entrée
    int A_cpu[SIZEA];
    int B_cpu[SIZEB];
    
	for (i = 0; i < SIZEA; i++)
		A_cpu[i] = 2 * i;
	
	for (i = 0; i < SIZEB; i++)
		B_cpu[i] = 2 * i + 1;
	
    // Allocation du tableau de sortie
	int M_cpu[SIZEA + SIZEB];			

    // Déclaration et allocation de la mémoire du GPU
	int *A_gpu, *B_gpu, *M_gpu, *Aindex, *Bindex;
	cudaMalloc( (void**) &A_gpu, SIZEA * sizeof(int) );
	cudaMalloc( (void**) &B_gpu, SIZEB * sizeof(int) );
	cudaMalloc( (void**) &M_gpu, (SIZEA+SIZEB) * sizeof(int) );
	cudaMalloc( (void**) &Aindex, (N_BLOCKS + 1) * sizeof(int) );
	cudaMalloc( (void**) &Bindex, (N_BLOCKS + 1) * sizeof(int) );

	// Copie des tableaux CPU vers GPU
	cudaMemcpy( A_gpu, A_cpu, SIZEA * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( B_gpu, B_cpu, SIZEB * sizeof(int), cudaMemcpyHostToDevice );

	// Kernel de partitionnement des tableaux par blocks
	pathBig_k<<<N_BLOCKS, 1>>>(A_gpu, B_gpu, Aindex, Bindex, SIZEA, SIZEB, N_BLOCKS);

	// Kernel de merge des partitions sur chaque block
	mergeBig_k<<<N_BLOCKS, N_THREADS>>>(A_gpu, B_gpu, M_gpu, Aindex, Bindex);

	// Copie du tableau résultat GPU vers CPU et affichage
	cudaMemcpy( M_cpu, M_gpu, (SIZEA+SIZEB) * sizeof(int), cudaMemcpyDeviceToHost );
	for (int i = 0; i < SIZEA+SIZEB; i ++)
		printf("M[%d] = %d\n", i, M_cpu[i]);
	
	// Liberation de la mémoire
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(M_gpu);
	cudaFree(Aindex);
	cudaFree(Bindex);

	return 0;
}
