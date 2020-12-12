 #include <stdlib.h>
#include <math.h>
#include <stdio.h>


#define N 1000
#define Nb 1200
#define NTPB 512
#define N_BLOCKS 4
#define X 0
#define Y 1
/*********************************************/
/************* PATH BIG K ********************/
/*********************************************/

__global__ void pathBig_k(int *a, int *b, int *Aindex, int *Bindex, int sizeA, int sizeB){
    // indexA et indexB sont stockés en mémoire globale
    // ils permettent de stocker les points de ruptures entre threads
	int K[2];
	int P[2];
	int Q[2];
	//int N_BLOCKS = (sizeA+sizeB+NTPB-1)/NTPB;
	int i = threadIdx.x + blockIdx.x * blockDim.x;    
	//int i = blockIdx.x+1;
   // int i = (sizeA + sizeB)/N_BLOCKS * (blockIdx.x + 1);
	//if (blockIdx.x == N_BLOCKS-1){
	//	return;
	//}
	//printf("i = %d\n",i);
    
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

        if(Q[1] >= 0 && Q[0] <= sizeB && (Q[1] == sizeA || Q[0] == 0 || a[ Q[1] ] > b[ Q[0]-1 ]) ){
            if(Q[0] == sizeB || Q[1] == 0 || a[ Q[1]-1 ] <= b[ Q[0] ]){
                Aindex[blockIdx.x] = Q[0];
                Bindex[blockIdx.x] = Q[1];
                printf("threadIdx: %d >>>>> blockIdx.x = %d | Aindex[%d] = %d | Bindex[%d] = %d \n", i, blockIdx.x, blockIdx.x, Q[0], blockIdx.x, Q[1]);
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

__global__ void mergeBig_k(int *A, int *B, int *M, int *A_idx, int *B_idx, int SIZEA, int SIZEB){

	// Mémoire shared sur laquelle nous allons travaillé
	extern __shared__ int A_shared[];
	extern __shared__ int B_shared[];

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
		endA = SIZEA;
		startB = B_idx[blockIdx.x-1];
		endB = SIZEB;
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
						printf("blockIdx.x = %d threadIdx.x = %d idx = %d m = %d biaisA = %d\n", blockIdx.x, threadIdx.x, idx, M[idx], biaisA);
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

int main(void){
    int sizeA = N;
    int sizeB = Nb;
    
	int *A;
	int *B;
	int *M;
	//int nb_blocks = (sizeA+sizeB+NTPB-1)/NTPB; // A vérifier

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

    cudaMalloc(&Aindex, N_BLOCKS * sizeof(int));
	cudaMalloc(&Bindex, N_BLOCKS * sizeof(int));

	cudaMemcpy(A_gpu, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
    
    pathBig_k<<<N_BLOCKS,1>>>(A_gpu, B_gpu, Aindex, Bindex, sizeA, sizeB);   
    mergeBig_k<<<N_BLOCKS,NTPB>>>(A_gpu, B_gpu, M_gpu, Aindex, Bindex, sizeA, sizeB);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("mergeBig_k: temps écoulé = %f secs\n", time/1000);


	cudaMemcpy(M, M_gpu, (N+Nb)*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N+Nb; i++)
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
