#include "../lib/headers/neuralnet.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(void) {

  srand(time(NULL));

  uint32_t result_code;
  const uint8_t num_operations = 6;

  pthread_t threads[num_operations];

  printf("\n");

  uint32_t i;

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
