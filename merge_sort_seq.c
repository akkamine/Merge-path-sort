#include <stdio.h>
#include <stdlib.h>

#define NSIZE(x) (sizeof(x) / sizeof((x)[0]))

int main(int argc, char const *argv[]) {
  int tab1[9] = {1,2,5,6,6,9,11,15,16};
  int tab2[7] = {4,7,8,10,12,13,14};
  size_t a = NSIZE(tab1);
  size_t b = NSIZE(tab2);
  int res[a+b];
  size_t m = NSIZE(res);

  int i=0,j=0;

  while (i+j < m) {
    if (i >= a){
      printf("1) i = %d j= %d\n",i,j);
      res[i+j] = tab2[j];
      j++;
    }
    else if (j >= b || tab1[i] < tab2[j]){
      printf("2) i = %d j= %d\n",i,j);
      res[i+j] = tab1[i];
      i++;
    }
    else{
      printf("3) i = %d j= %d\n",i,j);
      res[i+j] = tab2[j];
      j++;
    }
  }

  for(i=0; i<m; i++){
    printf("%d\n", res[i]);
  }


  return 0;
}
