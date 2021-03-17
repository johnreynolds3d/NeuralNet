#include "../lib/headers/neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));
  /*
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
  */

  struct Layer *layers[2];

  int i;

  for (i = 0; i < 2; i++) {
    layers[i] = Layer_create(8, 2);
    printf("layers[%d]->neurons[%d]->bias: %f\n", i, i,
           layers[i]->neurons[i]->bias);
    for (int j = 0; j < 8; j++) {
      printf("layers[%d]->neurons[%d]->weights[%d]: %f\n", i, i, j,
             layers[i]->neurons[i]->weights[j]);
      printf("layers[%d]->neurons[%d]->inputs[%d]: %f\n", i, i, j,
             layers[i]->neurons[i]->inputs[j]);
    }
  }
  /*
    for (i = 0; i < 2; i++) {
      Neuron_destroy(neurons[i]);
    }
    */

  for (i = 0; i < 2; i++) {
    Layer_destroy(layers[i]);
  }

  return 0;
}
