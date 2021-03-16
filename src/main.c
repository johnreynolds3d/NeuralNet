#include "../libs/headers/neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));

  struct Neuron *neurons[2];

  int i;

  for (i = 0; i < 2; i++) {
    neurons[i] = Neuron_create(4);
    printf("neurons[%d]->bias: %f\n", i, neurons[i]->bias);
    printf("neurons[%d]->weights[%d]: %f\n", i, i, neurons[i]->weights[i]);
  }

  for (i = 0; i < 2; i++) {
    Neuron_destroy(neurons[i]);
  }

  return 0;
}
