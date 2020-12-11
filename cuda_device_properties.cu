#include <stdio.h>

int main(void){
	// on déclare la structure qui servira de conteneur pour les informations qu'on requêtera
	cudaDeviceProp prop;
	// on fait notre requête au premier GPU (indice=0) pour avoir ses informations
	cudaGetDeviceProperties(&prop, 0);
	// on affiche ce qui nous intéresse dans ce qui a été récupéré
	printf("maxGridSize = (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("maxThreadsDim = (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
	// voir le type de structure "cudaDeviceProp" dans la doc pour d'autres champs...
}