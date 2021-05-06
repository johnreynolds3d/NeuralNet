#include "../lib/headers/neuralnet.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(void) {

  srand(time(NULL));

  int result_code;
  const int num_operations = 6;

  pthread_t threads[num_operations];

  printf("\n");

  int i;

  for (i = 0; i < num_operations; i++) {

    result_code = pthread_create(&threads[i], NULL, PreTraining, &i);
    assert(!result_code);
  }
  for (i = 0; i < num_operations; i++) {
    result_code = pthread_join(threads[i], NULL);
    assert(!result_code);
  }

  printf("\n\n");

  return 0;
}
