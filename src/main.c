#include "../libs/headers/neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));

  struct Neuron *neurons[2];

  int i;

  for (i = 0; i < 2; i++) {
    neurons[i] = Neuron_create(8);
    printf("neurons[%d]->bias: %f\n", i, neurons[i]->bias);
    for (int j = 0; j < 8; j++) {
      printf("neurons[%d]->weights[%d]: %f\n", i, j, neurons[i]->weights[j]);
      printf("neurons[%d]->inputs[%d]: %f\n", i, j, neurons[i]->inputs[j]);
    }
  }

  for (i = 0; i < 2; i++) {
    Neuron_destroy(neurons[i]);
  }

  return 0;
}
