#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define N 10000
#define NTPB 1024

__global__ void mergeSmallBatch_k(int *a, int *b, int *m, int *sizeA, int *sizeB, const int d){

	const int tidx = threadIdx.x % d;				//Num de la diagonal dans le tableau indice Qt
	const int Qt = (threadIdx.x - tidx) / d;			//Num du tableau dans le tableau shared
	const int gbx = Qt + blockIdx.x * (blockDim.x / d);		//Num du tableau dans le tableau global
	//printf("blockId.x = %d | threadIdx. x = %d | tidx = %d | Qt = %d | gbx = %d\n", blockIdx.x, threadIdx.x, tidx, Qt, gbx);
	
	//Taille du tableau en cours de traitement
	const int sizeAi = sizeA[gbx];				
	const int sizeBi = sizeB[gbx];				

	//Tableau partagé par les threads d'un bloc
	__shared__ int sA[1024];				
	__shared__ int sB[1024];				

	//Transfer des données dans la memeoir shared
	sA[Qt * d + tidx] = a[gbx * d + tidx];		
	sB[Qt * d + tidx] = b[gbx * d + tidx];		

	__syncthreads();					


	if (gbx * d + tidx >= N * d){
		return;
	}
	
	
	int K[2];
	int P[2];
	int Q[2];

	if (tidx > sizeAi) {
		K[0] = tidx - sizeAi;
		K[1] = sizeAi;
		P[0] = sizeAi;
		P[1] = tidx - sizeAi;
	}
	else{
		K[0] = 0; K[1] = tidx;
		P[0] = tidx; P[1] = 0;
	}

	while(1){

		int offset = (abs(K[1]-P[1]))/2;
		Q[0] = K[0] + offset;
		Q[1] = K[1] - offset;

		if(Q[1] >= 0 && Q[0] <= sizeBi && (Q[1] == sizeAi || Q[0] == 0 || sA[ Qt*d + Q[1] ] > sB[ Qt*d + Q[0]-1 ]) ){
		
			if(Q[0] == sizeBi || Q[1] == 0 || sA[Qt*d + Q[1]-1 ] <= sB[ Qt*d + Q[0] ]){
				
				if(Q[1] < sizeAi && (Q[0] == sizeBi || sA[Qt*d + Q[1] ] <= sB[ Qt*d + Q[0] ])){
	    	
	    			m[gbx * d + tidx] = sA[Qt*d + Q[1]] ;
	  			}
	  			else{
					m[gbx * d + tidx] = sB[Qt*d + Q[0]];
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
	int *A;
	int *B;
	int *M;

	const int d = 128;
	int nb_block = (N*d + NTPB-1)/NTPB;
	//printf("%d\n",nb_block);
	float time= 0.;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	A = (int*) malloc(N*d* sizeof(int));
	B = (int*) malloc(N*d* sizeof(int));
	M = (int*) calloc(sizeof(int),N*d);
	
	int * sizeAi = (int*) malloc(N * sizeof(int));
	int * sizeBi = (int*) malloc(N * sizeof(int));

	srand(0);
	
	for (int i = 0; i < N; i++){
		//int x;
		//printf("entrer la taille de A_%d\n",i);
		//scanf("%d",&x);
		
		// Taille aléatoire du tableau Ai[i]
		int x = rand() % d;

		sizeAi[i] = x;
		sizeBi[i] = d - x;

		// Remplissage des tableaux avec des valeurs croissantes car le tab doit etre trié
		for (int j = 0; j < sizeAi[i]; j ++){
			A[i*d+j] = 2*j;
		}
		for (int j = 0; j < sizeBi[i]; j ++){
			B[i*d+j] = 2*j + 1;
		}
	}
		
	int *A_gpu;
	int *B_gpu;
	int *M_gpu;
	int *sizeAi_GPU;
	int *sizeBi_GPU;
	
	cudaMalloc(&A_gpu, N*d* sizeof(int));
	cudaMalloc(&B_gpu, N*d* sizeof(int));
	cudaMalloc(&M_gpu, N*d* sizeof(int));
	cudaMalloc(&sizeAi_GPU, N * sizeof(int));
	cudaMalloc(&sizeBi_GPU, N * sizeof(int));

	cudaMemcpy(A_gpu, A, N*d* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, N*d* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(sizeAi_GPU, sizeAi, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(sizeBi_GPU, sizeBi, N * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	mergeSmallBatch_k<<<nb_block,NTPB>>>(A_gpu, B_gpu, M_gpu, sizeAi_GPU, sizeBi_GPU, d);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("mergeSmallBatch_k: temps écoulé = %f secs\n", time/1000);


	cudaMemcpy(M, M_gpu, N*d *sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < N+Nb; i++)
	//	printf("M[%d] = %d\n", i, M[i]);
	
	int i = rand()%N+1;
	printf("Tableau M _%d\n", i);
	if (sizeAi[i] != 0)
		printf("A_%d de size : %d | nb PAIRS allant 0 à %d\n",i, sizeAi[i], A[d*i+sizeAi[i]-1]);
	if (sizeBi[i] != 0)
		printf("B_%d de size : %d | nb IMPAIRS allant 1 à %d\n",i, sizeBi[i], B[d*i+sizeBi[i]-1]);


	for (int j = 0; j < d; j++){
		printf("M[%d][%d] = %d\n", i, j, M[i*d+j]);
	}

	

	free(A);
	free(B);
	free(M);
	free(sizeAi);
	free(sizeBi);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(M_gpu);
	cudaFree(sizeAi_GPU);
	cudaFree(sizeBi_GPU);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return 0;
}
